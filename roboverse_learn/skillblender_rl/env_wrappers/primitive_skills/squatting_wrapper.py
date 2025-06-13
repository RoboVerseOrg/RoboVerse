"""SkillBlench wrapper for training primitive skill: squatting."""

# ruff: noqa: F405
from __future__ import annotations

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.types import EnvState
from metasim.utils.humanoid_robot_util import *
from metasim.utils.math import sample_int_from_float
from roboverse_learn.skillblender_rl.env_wrappers.base.humanoid_base_wrapper import HumanoidBaseWrapper


class SquattingWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:walking

    # TODO implement push robot.
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        _, _ = self.env.reset()
        self._init_target_wp()

    def _parse_ref_root_height(self, envstate: EnvState):
        envstate.robots[self.robot.name].extra["ref_root_height"] = self.ref_root_height

    def _init_target_wp(self) -> None:
        self.target_wp, self.num_pairs, self.num_wp = self.sample_root_height(
            self.device,
            num_points=1000000,
            num_wp=10,
            ranges=self.cfg.command_ranges,
            base_height_target=self.cfg.reward_cfg.base_height_target,
        )  # relative, self.target_wp.shape=[num_pairs, num_wp, 1]
        self.target_wp_i = torch.randint(
            0, self.num_pairs, (self.num_envs,), device=self.device
        )  # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.human.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_wp_dt, (
            f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        )
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)

        self.ref_root_height = None
        self.ref_action = self.cfg.default_joint_pd_target
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.human.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))

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

        # NOTE: this line matters because target_wp should be reset before compute ons.
        self.update_target_wp(reset_env_idx)

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations(env_states)
        self._update_history(env_states)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

    def update_target_wp(self, reset_env_ids):
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_root_height = self.target_wp[self.target_wp_i, self.target_wp_j]  # [num_envs, 1]
        self.delayed_obs_target_wp = self.target_wp[
            self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))
        ]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.human.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(
            resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i
        )

    def _parse_state_for_reward(self, envstate: EnvState) -> None:
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        The

        Eg., offset the observation by default obs, compute input rewards.
        """
        # TODO read from config
        # parse those state which cannot directly get from Envstates
        super()._parse_state_for_reward(envstate)
        self._parse_ref_root_height(envstate)

    def _compute_observations(self, envstates: EnvState) -> None:
        """compute observation and privileged observation."""
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

        root_height = robot_position_tensor(envstates, self.robot.name)[:, 2].unsqueeze(1)
        ref_root_height = self.ref_root_height
        diff = root_height - self.ref_root_height  # [num_envs, 1]

        self.privileged_obs_buf = torch.cat(
            (
                ref_root_height,  # 1
                root_height,  # 1
                diff,  # 1
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
                diff,  # 3
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

    @staticmethod
    def sample_int_from_float(x: float) -> int:
        """Sample an int from a float."""
        if int(x) == x:
            return int(x)
        return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

    @staticmethod
    def sample_root_height(device, num_points, num_wp, ranges, base_height_target):
        """sample root height"""
        root_height = torch.randn(num_points, 1) * ranges.root_height_std + base_height_target
        root_height = torch.clamp(root_height, ranges.min_root_height, ranges.max_root_height)
        root_height = root_height.unsqueeze(1).repeat(1, num_wp, 1)  # (num_points, num_wp, 1)
        print("===> [sample_root_height] return shape:", root_height.shape)
        return root_height.to(device), num_points, num_wp
