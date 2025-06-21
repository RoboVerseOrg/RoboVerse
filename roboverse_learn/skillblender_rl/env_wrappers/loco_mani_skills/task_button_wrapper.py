"""SkillBlench wrapper for training loco-manipulation skill: ButtonPress."""

from __future__ import annotations

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.types import EnvState
from metasim.utils.humanoid_robot_util import (
    contact_forces_tensor,
    dof_pos_tensor,
    dof_vel_tensor,
)
from roboverse_learn.skillblender_rl.env_wrappers.base.humanoid_base_wrapper import HumanoidBaseWrapper


class TaskButtonWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:Button
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        _, _ = self.env.reset(self.init_states)
        self.env.handler.simulate()
        # prepare right arm indices for reward computation
        right_arm_joint_names = self.cfg.right_arm_joint_names
        self.cfg.right_shoulder_pitch_index = self.env.handler.get_joint_reindexed_indices_from_substring(
            self.robot.name, right_arm_joint_names
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.button_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        env_states = self.env.handler.get_states()
        self.wall_root_states = env_states.objects["wall"].root_state.clone()

    def _parse_button_goal_pos(self, envstate: EnvState):
        envstate.robots[self.robot.name].extra["button_goal_pos"] = self.button_goal_pos

    def _parse_state_for_reward(self, envstate: EnvState) -> None:
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """
        # TODO read from config
        # parse those state which cannot directly get from Envstates
        super()._parse_state_for_reward(envstate)
        self._parse_button_goal_pos(envstate)

    def _post_physics_step(self, env_states: EnvState) -> None:
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        # update episode length from env_wrapper
        self.episode_length_buf = self.env.episode_length_buf_tensor
        self.common_step_counter += 1

        self._post_physics_step_callback()
        # update refreshed tensors from simulaor
        self._update_refreshed_tensors(env_states)
        # prepare all the states for reward computation
        self._parse_state_for_reward(env_states)
        # compute the reward
        self.compute_reward(env_states)
        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        self.reset(reset_env_idx)

        # resampling botton goal
        self.button_goal_pos[reset_env_idx, 0] = self.wall_root_states[reset_env_idx, 0]
        self.button_goal_pos[reset_env_idx, 1] = self.wall_root_states[reset_env_idx, 1] + torch.FloatTensor(
            len(reset_env_idx)
        ).uniform_(*self.cfg.command_ranges.button_pos_y).to(self.device)
        self.button_goal_pos[reset_env_idx, 2] = self.cfg.button_ori_z + torch.FloatTensor(len(reset_env_idx)).uniform_(
            *self.cfg.command_ranges.button_pos_z
        ).to(self.device)

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations(env_states)
        self._update_history(env_states)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

    def _compute_observations(self, envstates: EnvState) -> None:
        """Add observation into states"""

        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel

        wrist_pose = envstates.robots[self.robot.name].body_state[:, self.wrist_indices, :7]
        wrist_pos = wrist_pose[:, 0, :3]  # [num_envs, 3], left hand, position only
        button_goal_pos = self.button_goal_pos[:, :3]  # [num_envs, 3]
        wrist_button_diff = wrist_pos - button_goal_pos  # [num_envs, 3], left hand, position only

        self.privileged_obs_buf = torch.cat(
            (
                button_goal_pos,  # 3
                wrist_pos,  # 3
                wrist_button_diff,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                wrist_button_diff,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )
