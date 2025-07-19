import torch


from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import learning.common_player as common_player

from rl_games.common.tr_helpers import unsqueeze_obs

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)

        # self.env.task.update_value_func(self._eval_critic, self._eval_actor)
        # import copy
        # self.orcale_model = copy.deepcopy(self.model)
        # checkpoint = torch_ext.load_checkpoint("output/dgx/smpl_im_master_singles_6_3/Humanoid_00031250.pth")
        # self.orcale_model.load_state_dict(checkpoint['model'])
        return

    # #### Oracle debug
    # def get_action(self, obs, is_determenistic=False):
    #     obs = obs['obs']
    #     if self.has_batch_dimension == False:
    #         obs = unsqueeze_obs(obs)
    #     obs = self._preproc_obs(obs)
    #     input_dict = {
    #         'is_train': False,
    #         'prev_actions': None,
    #         'obs': obs,
    #         'rnn_states': self.states
    #     }
    #     with torch.no_grad():
    #         res_dict = self.orcale_model(input_dict)
    #         print("orcale_model")

    #     mu = res_dict['mus']
    #     action = res_dict['actions']
    #     self.states = res_dict['rnn_states']
    #     if is_determenistic:
    #         current_action = mu
    #     else:
    #         current_action = action
    #     if self.has_batch_dimension == False:
    #         current_action = torch.squeeze(current_action.detach())

    #     if self.clip_actions:
    #         return rescale_actions(self.actions_low, self.actions_high,
    #                                torch.clamp(current_action, -1.0, 1.0))
    #     else:
    #         return current_action

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

            if self._normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        return

    def _eval_critic(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_task_value(input)


    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            
            # if self.env.task.has_task:
            #     config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
            #     config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
            #     config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _amp_debug(self, info):
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_obs(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_obs(self, obs_batch):
        
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            obs_batch_out = self.running_mean_std(obs_batch_proc)
            obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)
            
        return obs_batch
    

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r


class AMPPlayerDiscrete(common_player.CommonPlayerDiscrete):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)

        # self.env.task.update_value_func(self._eval_critic, self._eval_actor)
        # import copy
        # self.orcale_model = copy.deepcopy(self.model)
        # checkpoint = torch_ext.load_checkpoint("output/dgx/smpl_im_master_singles_6_3/Humanoid_00031250.pth")
        # self.orcale_model.load_state_dict(checkpoint['model'])
        return

    # #### Oracle debug
    # def get_action(self, obs, is_determenistic=False):
    #     obs = obs['obs']
    #     if self.has_batch_dimension == False:
    #         obs = unsqueeze_obs(obs)
    #     obs = self._preproc_obs(obs)
    #     input_dict = {
    #         'is_train': False,
    #         'prev_actions': None,
    #         'obs': obs,
    #         'rnn_states': self.states
    #     }
    #     with torch.no_grad():
    #         res_dict = self.orcale_model(input_dict)
    #         print("orcale_model")

    #     mu = res_dict['mus']
    #     action = res_dict['actions']
    #     self.states = res_dict['rnn_states']
    #     if is_determenistic:
    #         current_action = mu
    #     else:
    #         current_action = action
    #     if self.has_batch_dimension == False:
    #         current_action = torch.squeeze(current_action.detach())

    #     if self.clip_actions:
    #         return rescale_actions(self.actions_low, self.actions_high,
    #                                torch.clamp(current_action, -1.0, 1.0))
    #     else:
    #         return current_action

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

            if self._normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        return

    def _eval_critic(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_task_value(input)


    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
            if self.env.task.has_task:
                config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
                config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _amp_debug(self, info):
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_obs(self, obs_batch):
        
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            obs_batch_out = self.running_mean_std(obs_batch_proc)
            obs_batch = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)
            
        return obs_batch

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
