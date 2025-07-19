
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext

DISC_LOGIT_INIT_SCALE = 1.0


class AMPPNNBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPPNNBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.fut_tracks = self.task_obs_size_detail['fut_tracks']
            self.obs_v = self.task_obs_size_detail['obs_v']
            self.num_traj_samples = self.task_obs_size_detail['num_traj_samples']
            self.track_bodies = self.task_obs_size_detail['track_bodies']
            self.num_prim = self.task_obs_size_detail['num_prim']
            self.training_prim = self.task_obs_size_detail['training_prim']
            self.model_base = self.task_obs_size_detail['models_path'][0]
            self.actors_to_load = self.task_obs_size_detail['actors_to_load']
            self.has_lateral = self.task_obs_size_detail['has_lateral']

            kwargs['input_shape'] = (self.self_obs_size + self.task_obs_size,)  #

            super().__init__(params, **kwargs)
            actor_mlp_args = {
                'input_size': self._calc_input_size((self.self_obs_size + self.task_obs_size,), self.actor_cnn),
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }

            del self.actor_mlp
            self.discrete = params.get("discrete", False)

            self.pnn = PNN(actor_mlp_args, output_size=kwargs['actions_num'], numCols=self.num_prim, has_lateral=self.has_lateral)
            # self.pnn.load_base_net(self.model_base, self.actors_to_load)
            self.pnn.freeze_pnn(self.training_prim)

            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out, a_outs = self.pnn(a_out, idx=self.training_prim)

            # a_out = a_outs[0]
            # print("debugging")  # Dubgging!!!

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                # mu = self.mu_act(self.mu(a_out))
                mu = a_out
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return
