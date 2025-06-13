"""SkillBlench wrapper for training primitive skill: walking."""

# ruff: noqa: F405
from __future__ import annotations

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.humanoid_robot_util import *
from roboverse_learn.skillblender_rl.env_wrappers.base.humanoid_base_wrapper import HumanoidBaseWrapper


class WalkingWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:walking

    # TODO implement push robot
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        self._prepare_ref_indices()

    def _prepare_ref_indices(self):
        """get joint indices for reference pos computation."""
        joint_names = self.env.handler.get_joint_names(self.robot.name)
        self.left_hip_pitch_joint_idx = joint_names.index("left_hip_pitch")
        self.left_knee_joint_idx = joint_names.index("left_knee")
        self.left_ankle_joint_idx = joint_names.index("left_ankle")
        self.right_hip_pitch_joint_idx = joint_names.index("right_hip_pitch")
        self.right_knee_joint_idx = joint_names.index("right_knee")
        self.right_ankle_joint_idx = joint_names.index("right_ankle")

    def _compute_ref_state(self):
        """compute reference target position for walking task."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.env.handler.robot_num_dof, device=self.device, requires_grad=False
        )
        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1  # left_hip_pitch_joint
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2  # left_knee_joint
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1  # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1  # right_hip_pitch_joint
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2  # right_knee_joint
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1  # right_ankle_joint
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos

    def _parse_ref_pos(self, envstate):
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _parse_state_for_reward(self, envstate):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        The

        Eg., offset the observation by default obs, compute input rewards.
        """
        # TODO read from config
        # parse those state which cannot directly get from Envstates
        super()._parse_state_for_reward(envstate)
        self._compute_ref_state()
        self._parse_ref_pos(envstate)

    def _compute_observations(self, envstates):
        """compute observation and privileged observation."""

        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel
        diff = dof_pos_tensor(envstates, self.robot.name) - ref_dof_pos_tenosr(envstates, self.robot.name)

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff,  # |A|
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input_wo_clock,  # 3
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
