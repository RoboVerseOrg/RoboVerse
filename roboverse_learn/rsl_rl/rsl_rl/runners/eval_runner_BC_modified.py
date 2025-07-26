# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import statistics
import sys
import time

sys.path.append(os.getcwd())

import gc

import numpy as np
import torch
from tqdm import tqdm

from phc.smpllib.smpl_eval import compute_metrics_lite
from rsl_rl.env import VecEnv


class EvalRunnerBCModified:
    def __init__(self, env: VecEnv, policy, train_cfg, log_dir=None, device="cpu", clip_action=False, To=0):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.To = To
        self.policy = policy
        self.device = device
        self.policy.to(self.device)  # .half()
        self.env = env
        self.clip_actions = clip_action
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        # actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.success_count = 0
        self.failure_count = 0

        noise_vec = [
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.0100,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.1000,
            0.5000,
            0.5000,
            0.5000,
            0.1000,
            0.1000,
            0.1000,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
        ]
        self.noise_vec = torch.tensor(noise_vec).to(self.env.device)
        # TODO algin this hardcoding
        # if self.clip_actions:
        #     for i in range(len(self.env.dof_pos_limits)):
        #         soft_lower = self.env.dof_pos_limits[i, 0]
        #         soft_upper = self.env.dof_pos_limits[i, 1]
        #         m = (soft_lower + soft_upper) / 2
        #         r_soft = soft_upper - soft_lower
        #         soft_factor = 0.85

        #         r = r_soft / soft_factor
        #         lower = m - 0.5 * r
        #         upper = m + 0.5 * r
        #         self.env.dof_pos_limits[i, 0] = lower
        #         self.env.dof_pos_limits[i, 1] = upper
        #     self.env.dof_pos_limits[[4, 9],] *= 100

        # _, _ = self.env.reset()

    def update_training_data(self, failed_keys):
        humanoid_env = self.env
        humanoid_env._motion_lib.update_soft_sampling_weight(failed_keys)
        # joblib.dump({"failed_keys": failed_keys, "termination_history": humanoid_env._motion_lib._termination_history.clone()}, osp.join(self.network_path, f"failed_{self.epoch_num:010d}.pkl"))

    def eval(self):
        info = self.run_eval_loop()
        # if self.cfg.auto_negative_samping:
        #     self.update_training_data(info['failed_keys'])
        # del self.terminate_state, self.terminate_memory, self.mpjpe, self.mpjpe_all
        return info  # ["eval_info"]

    def run_eval_loop(self):
        print("############################ Evaluation ############################")

        self.env.begin_seq_motion_samples()
        self.env.compute_observations()
        # import ipdb;ipdb.set_trace()
        obs = self.env.obs_buf.clone()  # must get_observation() first, then we can get the latest obs_student_buf
        obs_window = []
        for _ in range(self.To):
            obs_window.append(obs)  # .half())
        obs_window_init = obs_window.copy()
        obs_input_init = torch.stack(obs_window_init, dim=1)
        obs_input = torch.stack(obs_window, dim=1)
        # import ipdb;ipdb.set_trace()
        self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        self.success_rate = 0
        self.pbar = tqdm(range(self.env._motion_lib._num_unique_motions // self.env.num_envs))
        self.pbar.set_description("")
        temp_max_distance = self.env.cfg.asset.termination_scales.max_ref_motion_distance
        self.env.cfg.env.test = True
        self.env.cfg.env.im_eval = True
        self.env.cfg.asset.termination_scales.max_ref_motion_distance = 1.0

        policy = self.policy
        # obs, privileged_obs = self.env.reset()
        batch_size = self.env.num_envs

        done_indices = []

        # Initialize data collection
        state_logs = {
            "rigid_body_pos": [],
            "rigid_body_xyzw_quat": [],
        }
        task_logs = {
            "target_positions": [],
            "target_quats_wxyz": [],
        }
        dones = torch.zeros(self.env.num_envs, dtype=torch.bool)
        timestep = 0
        terminate_step_counts = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        timeouts = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        BLENDER = True
        while True:
            obs_dict = {"obs": obs_input.detach()}  # .half()}
            start_time = time.time()
            action_dict = self.policy.predict_action(obs_dict)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Policy Execution time = {execution_time:.6f} seconds")
            actions = action_dict["action"][:, 0, :]
            if self.clip_actions:
                actions = (
                    torch.clip(
                        actions * 0.25 + self.env.env.handler.default_dof_pos,
                        # urdf order
                        self.env.env.handler.dof_pos_limits[:, 0],
                        self.env.env.handler.dof_pos_limits[:, 1],
                )   # urdf order
                    - self.env.env.handler.default_dof_pos
                ) * 4
            _, _, rews, dones, infos = self.env.step(actions.detach())
            timestep += 1
            if (dones & (terminate_step_counts == 0)).sum():
                pass
            timeouts = torch.where(
                dones & infos["time_outs"] & (terminate_step_counts == 0),
                torch.ones_like(timeouts, dtype=torch.bool),
                timeouts,
            )
            terminate_step_counts = torch.where(
                dones & (terminate_step_counts == 0),
                torch.full_like(terminate_step_counts, timestep),
                terminate_step_counts,
            )
            next_batch, check_end = self._post_step_eval(infos, timeouts, terminate_step_counts, timestep)
            obs_window.append(self.env.obs_buf)  # .half())
            if len(obs_window) > self.To:
                obs_window.pop(0)
            obs_input = torch.stack(obs_window, dim=1)

            for env_idx in range(self.env.num_envs):
                if dones[env_idx]:
                    current_obs = self.env.obs_buf[env_idx]
                    for t in range(self.To):
                        obs_window[t][env_idx] = current_obs
                    obs_input = torch.stack(obs_window, dim=1)

            if dones.sum() == self.env.num_envs:
                # import ipdb;ipdb.set_trace()
                self.env.reset()
                self.env.compute_observations()
                obs_reset = self.env.obs_buf.clone()
                obs_window.clear()
                for _ in range(self.To):
                    obs_window.append(obs_reset)
                obs_input = torch.stack(obs_window, dim=1)

            if next_batch:
                timestep = 0
                terminate_step_counts = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
                timeouts = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            if check_end["end"]:
                break
        # Save collected data
        anim_data = {"state_logs": state_logs, "task_logs": task_logs}

        save_path = os.path.join("./", "blender")
        os.makedirs(save_path, exist_ok=True)
        pickle_path = os.path.join(save_path, "animation.pkl")
        import pickle

        with open(pickle_path, "wb") as f:
            pickle.dump(anim_data, f)
        print(f"Saved animation data to: {pickle_path}")

        self.env.cfg.env.test = False
        self.env.cfg.env.im_eval = False
        self.env.cfg.asset.termination_scales.max_ref_motion_distance = temp_max_distance
        self.env.reset()  # Reset ALL environments, go back to training mode.

        torch.cuda.empty_cache()
        gc.collect()

        return check_end["eval_info"]

    def _post_step_eval(self, info, timeouts, terminate_step_counts, timesteps):
        end = False
        eval_info = {}
        # modify done such that games will exit and reset.
        humanoid_env = self.env

        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_stpes += 1
        next_batch = False

        if not (terminate_step_counts == 0).sum():
            self.terminate_memory.append(timeouts.cpu().numpy())
            self.success_count += timeouts.sum()
            self.failure_count += timeouts.shape[0] - timeouts.sum()
            self.success_rate = np.concatenate(self.terminate_memory)[
                : humanoid_env._motion_lib._num_unique_motions
            ].mean()
            all_mpjpe = torch.stack(self.mpjpe)
            all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(terminate_step_counts)]
            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(terminate_step_counts)]
            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(terminate_step_counts)]
            self.mpjpe_all.append(all_mpjpe)
            # import ipdb;ipdb.set_trace()

            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt
            # import ipdb;ipdb.set_trace()

            next_batch = True
            self.pbar.update(1)
            self.pbar.refresh()
            (
                self.mpjpe,
                self.gt_pos,
                self.pred_pos,
            ) = [], [], []

            # if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
            if humanoid_env.start_idx + humanoid_env.num_envs > humanoid_env._motion_lib._num_unique_motions:
                # import ipdb;ipdb.set_trace()
                self.pbar.clear()
                # import ipdb;ipdb.set_trace()
                terminate_hist = np.concatenate(self.terminate_memory)
                succ_idxes = np.flatnonzero(terminate_hist[: humanoid_env._motion_lib._num_unique_motions]).tolist()
                # import ipdb;ipdb.set_trace()
                pred_pos_all_succ = [
                    (self.pred_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes
                ]
                gt_pos_all_succ = [
                    (self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes
                ]

                pred_pos_all = self.pred_pos_all[: humanoid_env._motion_lib._num_unique_motions]
                gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                # humanoid_env._motion_lib.get_motion_num_steps().sum()

                # failed_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                # success_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                failed_keys = None
                success_keys = None
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])

                metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
                metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)
                # import ipdb;ipdb.set_trace()
                metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
                metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                if len(metrics_succ_print) == 0:
                    print("No success!!!")
                    metrics_succ_print = metrics_all_print

                print("------------------------------------------")
                print(f"Success Rate: {self.success_rate:.10f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                print("Succ: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
                # print("Failed keys: ", len(failed_keys), failed_keys)

                end = True

                eval_info = {
                    "eval_success_rate": self.success_rate,
                    "eval_mpjpe_all": metrics_all_print["mpjpe_g"],
                    "eval_mpjpe_succ": metrics_succ_print["mpjpe_g"],
                    "accel_dist": metrics_succ_print["accel_dist"],
                    "vel_dist": metrics_succ_print["vel_dist"],
                    "mpjpel_all": metrics_all_print["mpjpe_l"],
                    "mpjpel_succ": metrics_succ_print["mpjpe_l"],
                    "mpjpe_pa": metrics_succ_print["mpjpe_pa"],
                }

                return next_batch, {
                    "end": end,
                    "eval_info": eval_info,
                    "failed_keys": failed_keys,
                    "success_keys": success_keys,
                }
            humanoid_env.forward_motion_samples()

        update_str = f"Terminated: {self.failure_count} | /*max frames: {humanoid_env._motion_lib.get_motion_num_steps().max()} | steps {timesteps} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return next_batch, {"end": end, "eval_info": eval_info, "failed_keys": [], "success_keys": []}

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar(
            "Episode/Average_episode_length_for_reward_curriculum",
            locs["average_episode_length_for_reward_curriculum"],
            locs["it"],
        )
        self.writer.add_scalar("Episode/Penalty_scale", locs["penalty_scale"], locs["it"])
        self.writer.add_scalar(
            "Episode/Teleop_body_pos_upperbody_sigma", locs["teleop_body_pos_upperbody_sigma"], locs["it"]
        )
        self.writer.add_scalar("Episode/Born_distance", locs["born_distance"], locs["it"])
        self.writer.add_scalar("Episode/Born_heading", locs["born_heading"], locs["it"])

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/action_smoothness", locs["mean_action_smoothness_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_cost", statistics.mean(locs["costbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_cost/time", statistics.mean(locs["costbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        if "eval_info" in locs:
            self.writer.add_scalar("Eval/Success_rate", locs["eval_info"]["eval_success_rate"], locs["it"])
            self.writer.add_scalar("Eval/Mpjpe_all", locs["eval_info"]["eval_mpjpe_all"], locs["it"])
            self.writer.add_scalar("Eval/Mpjpe_succ", locs["eval_info"]["eval_mpjpe_succ"], locs["it"])
            self.writer.add_scalar("Eval/Accel_dist", locs["eval_info"]["accel_dist"], locs["it"])
            self.writer.add_scalar("Eval/Vel_dist", locs["eval_info"]["vel_dist"], locs["it"])
            self.writer.add_scalar("Eval/Mpjpel_all", locs["eval_info"]["mpjpel_all"], locs["it"])
            self.writer.add_scalar("Eval/Mpjpel_succ", locs["eval_info"]["mpjpel_succ"], locs["it"])
            self.writer.add_scalar("Eval/Mpjpe_pa", locs["eval_info"]["mpjpe_pa"], locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Action smoothness loss:":>{pad}} {locs["mean_action_smoothness_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
                f"""{"Mean cost:":>{pad}} {statistics.mean(locs["costbuffer"]):.2f}\n"""
                f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
                f"""{"Average_episode_length_for_reward_curriculum:":>{pad}} {locs["average_episode_length_for_reward_curriculum"]:.6f}\n"""
                f"""{"Born_distance:":>{pad}} {locs["born_distance"]:.6f}\n"""
                f"""{"Born_heading:":>{pad}} {locs["born_heading"]:.6f}\n"""
                f"""{"Penalty_scale:":>{pad}} {locs["penalty_scale"]:.6f}\n"""
                f"""{"Teleop_body_pos_upperbody_sigma:":>{pad}} {locs["teleop_body_pos_upperbody_sigma"]:.6f}\n"""
            )
        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Action smoothness loss:":>{pad}} {locs["mean_action_smoothness_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        if "mean_kin_loss" in locs:
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Mean kin loss:":>{pad}} {locs["mean_kin_loss"]:.3f}\n"""
            )

        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {
                self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"]):.1f}s\n"""
        )

        log_string += f"""path: {self.log_dir}\n"""
        print("\r " + log_string, end="")

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
