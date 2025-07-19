

import torch
import torch.nn as nn
from phc.learning.network_builder import NetworkBuilder
from collections import defaultdict
from rl_games.algos_torch import torch_ext
from tqdm import tqdm


class PNN(NetworkBuilder.BaseNetwork):

    def __init__(self, mlp_args, output_size=69, numCols=4, has_lateral=True):
        super(PNN, self).__init__()
        self.numCols = numCols
        units = mlp_args['units']
        dense_func = mlp_args['dense_func']
        self.has_lateral = has_lateral

        self.actors = nn.ModuleList()
        for i in range(numCols):
            mlp = self._build_sequential_mlp(output_size, **mlp_args)
            self.actors.append(mlp)

        if self.has_lateral:

            self.u = nn.ModuleList()

            for i in range(numCols - 1):
                self.u.append(nn.ModuleList())
                for j in range(i + 1):
                    u = nn.Sequential()
                    in_size = units[0]
                    for unit in units[1:]:
                        u.append(dense_func(in_size, unit, bias=False))
                        in_size = unit
                    u.append(dense_func(units[-1], output_size, bias=False))
                    #                     torch.nn.init.zeros_(u[-1].weight)
                    self.u[i].append(u)

    def freeze_pnn(self, idx):
        for param in self.actors[:idx].parameters():
            param.requires_grad = False
        if self.has_lateral:
            for param in self.u[:idx - 1].parameters():
                param.requires_grad = False

    def load_base_net(self, model_path, actors=1):
        checkpoint = torch_ext.load_checkpoint(model_path)
        for idx in range(actors):
            self.load_actor(checkpoint, idx)

    def load_actor(self, checkpoint, idx=0):
        state_dict = self.actors[idx].state_dict()
        state_dict['0.weight'].copy_(checkpoint['model']['a2c_network.actor_mlp.0.weight'])
        state_dict['0.bias'].copy_(checkpoint['model']['a2c_network.actor_mlp.0.bias'])
        state_dict['2.weight'].copy_(checkpoint['model']['a2c_network.actor_mlp.2.weight'])
        state_dict['2.bias'].copy_(checkpoint['model']['a2c_network.actor_mlp.2.bias'])
        state_dict['4.weight'].copy_(checkpoint['model']['a2c_network.mu.weight'])
        state_dict['4.bias'].copy_(checkpoint['model']['a2c_network.mu.bias'])

    def _build_sequential_mlp(self, actions_num, input_size, units, activation, dense_func, norm_only_first_layer=False, norm_func_name=None, need_norm = True):
        print('build mlp:', input_size)
        in_size = input_size
        layers = []
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(self.activations_factory.create(activation))
            
            if not need_norm:
                continue
            if norm_only_first_layer and norm_func_name is not None:
                need_norm = False
            if norm_func_name == 'layer_norm':
                layers.append(torch.nn.LayerNorm(unit))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm1d(unit))
            in_size = unit
            

        layers.append(nn.Linear(units[-1], actions_num))
        return nn.Sequential(*layers)

    def forward(self, x, idx=-1):
        if self.has_lateral:
            # idx == -1: forward all, output all
            # idx == others, forward till idx.
            if idx == 0:
                actions = self.actors[0](x)
                return actions, [actions]
            else:
                if idx == -1:
                    idx = self.numCols - 1
                activation_cache = defaultdict(list)

                for curr_idx in range(0, idx + 1):
                    curr_actor = self.actors[curr_idx]
                    assert len(curr_actor) == 5  # Only support three MLPs right now
                    activation_1 = curr_actor[:2](x)

                    acc_acts_1 = [self.u[curr_idx - 1][col_idx][0](activation_cache[0][col_idx]) for col_idx in range(len(activation_cache[0]))]  # curr_idx - 1 as we need to go to the previous coloumn's index to activate the weight
                    activation_2 = curr_actor[3](curr_actor[2](activation_1) + sum(acc_acts_1))  # ReLU, full

                    # acc_acts_2 = [self.u[curr_idx - 1][col_idx][1](activation_cache[1][col_idx]) for col_idx in range(len(activation_cache[1]))]
                    # actions = curr_actor[4](activation_2) + sum(acc_acts_2)

                    actions = curr_actor[4](activation_2)  # disable action space transfer.

                    #                     acc_acts_1 = []
                    #                     for col_idx in range(len(activation_cache[0])):
                    #                         acc_acts_1.append(self.u[curr_idx - 1][col_idx][0](activation_cache[0][col_idx]))

                    #                     activation_2 = curr_actor[3](curr_actor[2](activation_1) + sum(acc_acts_1))

                    #                     acc_acts_2 = []
                    #                     for col_idx in range(len(activation_cache[1])):
                    #                         acc_acts_2.append(self.u[curr_idx - 1][col_idx][1](activation_cache[1][col_idx]))
                    #                     actions = curr_actor[4](activation_2) + sum(acc_acts_2)

                    activation_cache[0].append(activation_1)
                    activation_cache[1].append(activation_2)
                    activation_cache[2].append(actions)

                return actions, activation_cache[2]
        else:
            if idx != -1:
                actions = self.actors[idx](x)
                return actions, [actions]
            else:
                actions = [self.actors[idx](x) for idx in range(self.numCols)]
                return actions, actions
