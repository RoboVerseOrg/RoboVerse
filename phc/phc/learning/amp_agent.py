from phc.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn
from phc.env.tasks.humanoid_amp_task import HumanoidAMPTask

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent

from tensorboardX import SummaryWriter
import copy
from phc.utils.torch_utils import project_to_norm
import learning.amp_datasets as amp_datasets
from phc.learning.loss_functions import kl_multi
from uhc.utils.math_utils import LinearAnneal

def load_my_state_dict(target, saved_dict):
    for name, param in saved_dict.items():
        if name not in target:
            continue

        if target[name].shape == param.shape:
            target[name].copy_(param)


class AMPAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)


        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)  # Override and get new value

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        norm_disc_reward = config.get('norm_disc_reward', False)
        if (norm_disc_reward):
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        self.temp_running_mean = self.vec_env.env.task.temp_running_mean # use temp running mean to make sure the obs used for training is the same as calc gradient.

        kin_lr = float(self.vec_env.env.task.kin_lr)
        
        # ZL Hack
        if self.vec_env.env.task.fitting:
            print("#################### Fitting and freezing!! ####################")
            checkpoint = torch_ext.load_checkpoint(self.vec_env.env.task.models_path[0])
            
            self.set_stats_weights(checkpoint)  # loads mean std. essential for distilling knowledge. will not load if has a shape mismatch.
            self.freeze_state_weights()  # freeze the mean stds.
            load_my_state_dict(self.model.state_dict(), checkpoint['model'])  # loads everything (model, std, ect.). that can be load from the last model.
            # self.value_mean_std # not freezing value function though.
        
        return
    
    def set_stats_weights(self, weights):
        if self.normalize_input:
            if weights['running_mean_std']['running_mean'].shape == self.running_mean_std.state_dict()['running_mean'].shape:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
            else:
                print("shape mismatch, can not load input mean std")
                
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])

        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
 
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])
            
        if self._normalize_amp_input:
            if weights['amp_input_mean_std']['running_mean'].shape == self._amp_input_mean_std.state_dict()['running_mean'].shape:
                self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
            else:
                print("shape mismatch, can not load AMP mean std")
            

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.load_state_dict(weights['disc_reward_mean_std'])
            
    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        
        if "kin_optimizer" in self.__dict__:
            print("!!!saving kin_optimizer!!! Remove this message asa p!!")
            state['kin_optimizer'] = self.kin_optimizer.state_dict()

        return state

    def set_full_state_weights(self, weights):
        super().set_full_state_weights(weights)
        if "kin_optimizer" in weights:
            print("!!!loading kin_optimizer!!! Remove this message asa p!!")
            self.kin_optimizer.load_state_dict(weights['kin_optimizer'])
        

    def freeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.freeze()
        if self.normalize_value:
            self.value_mean_std.freeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def unfreeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.unfreeze()
        if self.normalize_value:
            self.value_mean_std.unfreeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()

            
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.eval()

        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.train()

        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()

        if (self._norm_disc_reward()):
            state['disc_reward_mean_std'] = self._disc_reward_mean_std.state_dict()

        return state
    

    def play_steps_rnn(self):
        self.set_eval()
        mb_rnn_states = []
        epinfos = []
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)
        step_time = 0.0

        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states) # mb_rnn_states means "memory bank" rnn states

        ### ZL
        done_indices = []
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)

        for n in range(self.horizon_length):
            
            
            
            self.obs = self.env_reset(done_indices)
            
            # self.rnn_states[0][:, :, -1] = n; print('debugg!!!!')
            # self.rnn_states[0][:, :, -2] = torch.arange(self.num_actors)
            
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)  # this should upate mb_rnn_states
            if full_tensor:
                break
            
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data_rnn('obses', indices, play_mask, self.obs['obs'])

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn('states', indices[::self.num_agents], play_mask[::self.num_agents] // self.num_agents, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
                
            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn('next_obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones', indices, play_mask, self.dones.byte())
            self.experience_buffer.update_data_rnn('amp_obs', indices, play_mask, infos['amp_obs'])

            ### ZL
            terminated = infos['terminate'].float()
            terminated_flags += terminated
            reward_raw_mean = infos['reward_raw'].mean(dim=0)

            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            terminated = terminated.unsqueeze(-1)
            input_dict = {"obs": self.obs['obs'], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)  # ZL this has issues? (maybe not, since we are passing the states in.)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data_rnn('next_values', indices, play_mask, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)

            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]
            

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        
        # self.experience_buffer.tensor_dict['actions']: is num_env, Batch, feat. That's why we swap and flatten, mb_rnn_states is already in that format. 
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # swap to step, num_envs, feat
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['rnn_states'] = mb_rnn_states
        
        batch_dict['rnn_masks'] = mb_rnn_masks # ZL: this should be swap and flattened, but it's all ones for now
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        
        batch_dict['played_frames'] = n * self.num_actors * self.num_agents
        batch_dict['step_time'] = step_time
        

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def play_steps(self):
        self.set_eval()
        humanoid_env = self.vec_env.env.task

        epinfos = []
        done_indices = []
        update_list = self.update_list
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

                
            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        batch_dict['played_frames'] = self.batch_size
        
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def prepare_dataset(self, batch_dict):
        
        
        dataset_dict = super().prepare_dataset(batch_dict)
        dataset_dict['amp_obs'] = batch_dict['amp_obs']
        dataset_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        dataset_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']

            
        self.dataset.update_values_dict(dataset_dict, rnn_format = True, horizon_length = self.horizon_length, num_envs = self.num_actors)
        # self.dataset.update_values_dict(dataset_dict)

        return

    def train_epoch(self):
        self.pre_epoch(self.epoch_num)
        play_time_start = time.time()

        ### ZL: do not update state weights during play

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        # if self.is_rnn:
        # frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        train_info['terminated_flags'] = batch_dict['terminated_flags']
        train_info['reward_raw'] = batch_dict['reward_raw']
        train_info['mb_rewards'] = batch_dict['mb_rewards']
        train_info['returns'] = batch_dict['returns']
        self._record_train_batch_info(batch_dict, train_info)
        self.post_epoch(self.epoch_num)
        
        return train_info

    def pre_epoch(self, epoch_num):
        # print("freeze running mean/std")

        if self.vec_env.env.task.smpl_humanoid:
            humanoid_env = self.vec_env.env.task
            if (epoch_num > 1) and epoch_num % humanoid_env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            # if (epoch_num > 0) and epoch_num % humanoid_env.shape_resampling_interval == 0 and not (epoch_num % (self.save_freq)): # Remove the resampling for this. 
                # Different from AMP, always resample motion no matter the motion type.
                print("Resampling Shape")
                humanoid_env.resample_motions()
                # self.current_rewards # Fixing these values such that they do not get whacked by the
                # self.current_lengths
            if humanoid_env.getup_schedule:
                humanoid_env.update_getup_schedule(epoch_num, getup_udpate_epoch=humanoid_env.getup_udpate_epoch)
                if epoch_num > humanoid_env.getup_udpate_epoch:  # ZL fix janky hack
                    self._task_reward_w = 0.5
                    self._disc_reward_w = 0.5
                else:
                    self._task_reward_w = 0
                    self._disc_reward_w = 1

        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Freeze running mean/std, so that the actor does not use the updated mean/std
        self.running_mean_std_temp.freeze()

    def post_epoch(self, epoch_num):
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Unfreeze running mean/std
        self.running_mean_std_temp.freeze()
        

    def _preproc_obs(self, obs_batch, use_temp=False):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v, use_temp = use_temp)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            if use_temp:
                obs_batch_out = self.running_mean_std_temp(obs_batch_proc)
                obs_batch_orig = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            else:
                obs_batch_out = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    def calc_gradients(self, input_dict):
        
        self.set_train()
        humanoid_env = self.vec_env.env.task

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch_processed = self._preproc_obs(obs_batch, use_temp=self.temp_running_mean)
        input_dict['obs_processed'] = obs_batch_processed

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        
        self.train_result = {}
        
        batch_dict = {'is_train': True, 'amp_steps': self.vec_env.env.task._num_amp_obs_steps, \
            'prev_actions': actions_batch, 'obs': obs_batch_processed, 'amp_obs': amp_obs, 'amp_obs_replay': amp_obs_replay, 'amp_obs_demo': amp_obs_demo, \
                "obs_orig": obs_batch
                }
    
        rnn_masks = None
        rnn_len = self.horizon_length
        rnn_len = 1
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = rnn_len
            
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict) # current model if RNN, has BPTT enabled. 
            
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            if not rnn_masks is None:
                rnn_mask_bool = rnn_masks.squeeze().bool()
                old_action_log_probs_batch, action_log_probs, advantage, values, entropy, mu, sigma, return_batch, old_mu_batch, old_sigma_batch = \
                    old_action_log_probs_batch[rnn_mask_bool], action_log_probs[rnn_mask_bool], advantage[rnn_mask_bool], values[rnn_mask_bool], \
                        entropy[rnn_mask_bool], mu[rnn_mask_bool], sigma[rnn_mask_bool], return_batch[rnn_mask_bool], old_mu_batch[rnn_mask_bool], old_sigma_batch[rnn_mask_bool]
                
                # flatten values for computing loss
                
                
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                + self._disc_coef * disc_loss
            
            
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = kl_dist.mean()
        
                
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
          
        self.train_result.update( {'entropy': entropy, 'kl': kl_dist, 'last_lr': self.last_lr, 'lr_mul': lr_mul, 'b_loss': b_loss})
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
            
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert (self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        
        config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
        if self.vec_env.env.task.has_task:
            config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
            config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return


    def _oracle_loss(self, obs):
        oracle_a, _ = self.oracle_model.a2c_network.eval_actor({"obs": obs})
        model_a, _ = self.model.a2c_network.eval_actor({"obs": obs})
        oracle_loss = (oracle_a - model_a).pow(2).mean(dim=-1) * 50
        return {'oracle_loss': oracle_loss}

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        '''
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit 
        obs_demo: gradient penalty demo obs (real examples)
        '''
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights)) # make weight small??
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]

        ### ZL Hack for zeroing out gradient penalty on the shape (406,)
        # if self.vec_env.env.task.__dict__.get("smpl_humanoid", False):
        #     humanoid_env = self.vec_env.env.task
        #     B, feat_dim = disc_demo_grad.shape
        #     shape_obs_dim = 17
        #     if humanoid_env.has_shape_obs:
        #         amp_obs_dim = int(feat_dim / humanoid_env._num_amp_obs_steps)
        #         for i in range(humanoid_env._num_amp_obs_steps):
        #             disc_demo_grad[:,
        #                            ((i + 1) * amp_obs_dim -
        #                             shape_obs_dim):((i + 1) * amp_obs_dim)] = 0

        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        # print(f"agent_loss: {disc_loss_agent.item():.3f}  | disc_loss_demo {disc_loss_demo.item():.3f}")
        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape, device=self.ppo_device)
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)  # Demo is the data from the dataset. Real samples

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return

    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _norm_disc_reward(self):
        return self._disc_reward_mean_std is not None

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']

        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {'disc_rewards': disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))

            if (self._norm_disc_reward()):
                self._disc_reward_mean_std.train()
                norm_disc_r = self._disc_reward_mean_std(disc_r.flatten())
                disc_r = norm_disc_r.reshape(disc_r.shape)
                disc_r = 0.5 * disc_r + 0.25

            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = super()._assemble_train_info(train_info, frame)
        
        if "disc_loss" in train_info:
            disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
            train_info_dict.update({
                "disc_loss": torch_ext.mean_list(train_info['disc_loss']).item(),
                "disc_agent_acc": torch_ext.mean_list(train_info['disc_agent_acc']).item(),
                "disc_demo_acc": torch_ext.mean_list(train_info['disc_demo_acc']).item(),
                "disc_agent_logit": torch_ext.mean_list(train_info['disc_agent_logit']).item(),
                "disc_demo_logit": torch_ext.mean_list(train_info['disc_demo_logit']).item(),
                "disc_grad_penalty": torch_ext.mean_list(train_info['disc_grad_penalty']).item(),
                "disc_logit_loss": torch_ext.mean_list(train_info['disc_logit_loss']).item(),
                "disc_reward_mean": disc_reward_mean.item(),
                "disc_reward_std": disc_reward_std.item(),
            })
        
        if "returns" in train_info:
            train_info_dict['returns'] = train_info['returns'].mean().item()
            
        if "mb_rewards" in train_info:
            train_info_dict['mb_rewards'] = train_info['mb_rewards'].mean().item()
        
        if 'terminated_flags' in train_info:
            train_info_dict["success_rate"] =  1 - torch.mean((train_info['terminated_flags'] > 0).float()).item()
        
        if "reward_raw" in train_info:
            for idx, v in enumerate(train_info['reward_raw'].cpu().numpy().tolist()):
                train_info_dict[f"ind_reward.{idx}"] =  v
        
        if "sym_loss" in train_info:
            train_info_dict['sym_loss'] = torch_ext.mean_list(train_info['sym_loss']).item()
        return train_info_dict

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)
        return
