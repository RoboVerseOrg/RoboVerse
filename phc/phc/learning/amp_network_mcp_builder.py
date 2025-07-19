
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy

DISC_LOGIT_INIT_SCALE = 1.0


class AMPMCPBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPMCPBuilder.Network(self.params, **kwargs)
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
            self.has_softmax = params.get("has_softmax", True)

            kwargs['input_shape'] = (self.self_obs_size + self.task_obs_size,)  #

            super().__init__(params, **kwargs)

            self.num_primitive = self.task_obs_size_detail.get("num_prim", 4)

            composer_mlp_args = {
                'input_size': self._calc_input_size((self.self_obs_size + self.task_obs_size,), self.actor_cnn),
                'units': self.units + [self.num_primitive],
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }

            self.composer = self._build_mlp(**composer_mlp_args)
            
            if self.has_softmax:
                print("!!!Has softmax!!!")
                self.composer.append(nn.Softmax(dim=1))

            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

        def load(self, params):
            super().load(params)
            return

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            a_out = self.composer(a_out)

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
