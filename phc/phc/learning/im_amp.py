

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

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
import phc.learning.amp_agent as amp_agent
from phc.utils.flags import flags
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions

from tensorboardX import SummaryWriter
import joblib
import gc
from uhc.smpllib.smpl_eval import compute_metrics_lite
from tqdm import tqdm


class IMAmpAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        

    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict["obs"]

        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def env_eval_step(self, env, actions):

        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

    def restore(self, fn):
        super().restore(fn)
        
        all_fails = glob.glob(osp.join(self.network_path, f"failed_*"))
        if len(all_fails) > 0:
            print("------------------------------------------------------ Restoring Termination History ------------------------------------------------------")
            failed_pth = sorted(all_fails, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"loading: {failed_pth}")
            termination_history = joblib.load(failed_pth)['termination_history']
            humanoid_env = self.vec_env.env.task
            res = humanoid_env._motion_lib.update_sampling_prob(termination_history)
            if res:
                print("Successfully restored termination history")
            else:
                print("Termination history length does not match")
            
        return
    
    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.vec_env.env.task.num_envs, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]
            
            
    def update_training_data(self, failed_keys):
        humanoid_env = self.vec_env.env.task
        joblib.dump({"failed_keys": failed_keys, "termination_history": humanoid_env._motion_lib._termination_history}, osp.join(self.network_path, f"failed_{self.epoch_num:010d}.pkl"))
        
        
        
    def eval(self):
        print("############################ Evaluation ############################")
        if not flags.has_eval:
            return {}

        self.set_eval()

        self.terminate_state = torch.zeros(
            self.vec_env.env.task.num_envs, device=self.device
        )
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        humanoid_env = self.vec_env.env.task
        self.success_rate = 0
        self.pbar = tqdm(
            range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs)
        )
        self.pbar.set_description("")

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        termination_distances, cycle_motion, zero_out_far, reset_ids = (
            humanoid_env._termination_distances.clone(),
            humanoid_env.cycle_motion,
            humanoid_env.zero_out_far,
            humanoid_env._reset_bodies_id,
        )

        if "_recovery_episode_prob" in humanoid_env.__dict__:
            recovery_episode_prob, fall_init_prob = (
                humanoid_env._recovery_episode_prob,
                humanoid_env._fall_init_prob,
            )
            humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        humanoid_env._termination_distances[:] = 0.5  # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env.cycle_motion = False
        humanoid_env.zero_out_far = False
        flags.test, flags.im_eval = (True, True,)  # need to be test to have: motion_times[:] = 0
        humanoid_env._motion_lib = humanoid_env._motion_eval_lib
        humanoid_env.begin_seq_motion_samples()
        if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################

        self.print_stats = False
        self.has_batch_dimension = True

        need_init_rnn = self.is_rnn
        obs_dict = self.env_reset()
        batch_size = humanoid_env.num_envs

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        done_indices = []

        with torch.no_grad():
            while True:
                obs_dict = self.env_reset(done_indices)

                action = self.get_action(obs_dict, is_determenistic=True)
                obs_dict, r, done, info = self.env_eval_step(self.vec_env.env, action)
                cr += r
                steps += 1
                done, info = self._post_step_eval(info, done.clone())

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                    done_indices = done_indices[:, 0]

                if info['end']:
                    break

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        humanoid_env._termination_distances[:] = termination_distances
        humanoid_env.cycle_motion = cycle_motion
        humanoid_env.zero_out_far = zero_out_far
        flags.test, flags.im_eval = False, False
        humanoid_env._motion_lib = humanoid_env._motion_train_lib
        if "_recovery_episode_prob" in humanoid_env.__dict__:
            humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = (
                recovery_episode_prob,
                fall_init_prob,
            )
        humanoid_env._reset_bodies_id = reset_ids
        self.env_reset()  # Reset ALL environments, go back to training mode.

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        torch.cuda.empty_cache()
        gc.collect()
        
        self.update_training_data(info['failed_keys'])
        del self.terminate_state, self.terminate_memory, self.mpjpe, self.mpjpe_all
        return info["eval_info"]

    def _post_step_eval(self, info, done):
        end = False
        eval_info = {}
        # modify done such that games will exit and reset.
        humanoid_env = self.vec_env.env.task
        termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
        # termination_state = info["terminate"]
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
        if (~self.terminate_state).sum() > 0:
            max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
            curr_ids = humanoid_env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][
                        ~self.terminate_state[:bound]
                    ].max()
                else:
                    curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()
                
            if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
        else:
            curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_stpes += 1

        if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
            self.curr_stpes = 0
            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.success_rate = (1- np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
            all_mpjpe = [all_mpjpe[:(i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred = [all_body_pos_pred[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt = [all_body_pos_gt[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]


            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt
            

            if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                self.pbar.clear()
                terminate_hist = np.concatenate(self.terminate_memory)
                succ_idxes = np.flatnonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]).tolist()

                pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]


                # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                # humanoid_env._motion_lib.get_motion_num_steps().sum()

                failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])

                metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
                metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
                metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()}
                
                if len(metrics_succ_print) == 0:
                    print("No success!!!")
                    metrics_succ_print = metrics_all_print
                    
                print("------------------------------------------")
                print(f"Success Rate: {self.success_rate:.10f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
                print("Failed keys: ", len(failed_keys), failed_keys)
                
                end = True
                
                eval_info = {
                    "eval_success_rate": self.success_rate,
                    "eval_mpjpe_all": metrics_all_print['mpjpe_g'],
                    "eval_mpjpe_succ": metrics_succ_print['mpjpe_g'],
                    "accel_dist": metrics_succ_print['accel_dist'], 
                    "vel_dist": metrics_succ_print['vel_dist'], 
                    "mpjpel_all": metrics_all_print['mpjpe_l'],
                    "mpjpel_succ": metrics_succ_print['mpjpe_l'],
                    "mpjpe_pa": metrics_succ_print['mpjpe_pa'], 
                }
                # failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                # joblib.dump(failed_keys, "output/dgx/smpl_im_shape_long_1/failed_1.pkl")
                # joblib.dump(success_keys, "output/dgx/smpl_im_fit_3_1/long_succ.pkl")
                # print("....")
                return done, {"end": end, "eval_info": eval_info, "failed_keys": failed_keys,  "success_keys": success_keys}

            done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

            humanoid_env.forward_motion_samples()
            self.terminate_state = torch.zeros(self.vec_env.env.task.num_envs, device=self.device)

            self.pbar.update(1)
            self.pbar.refresh()
            self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []


        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return done, {"end": end, "eval_info": eval_info, "failed_keys": [],  "success_keys": []}
