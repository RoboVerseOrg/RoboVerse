"""SkillBlench wrapper for training loco-manipulation Skillbench:PackageLift"""

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


class TaskLiftWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:PackageLift
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        _, _ = self.env.reset(self.init_states)
        self.env.handler.simulate(self.init_states)

    def _init_buffers(self):
        super()._init_buffers()
        self.box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _parse_box_goal_pos(self, envstate: EnvState):
        """Parse box goal position from envstate"""
        envstate.objects["box"].extra = {}
        envstate.objects["box"].extra["box_goal_pos"] = self.box_goal_pos

    def _parse_state_for_reward(self, envstate: EnvState) -> None:
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """
        super()._parse_state_for_reward(envstate)
        self._parse_box_goal_pos(envstate)

    def _compute_observations(self, envstates: EnvState) -> None:
        """Add observation into states"""

        # humanoid observations
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

        # box observations
        wrist_pos = envstates.robots[self.robot.name].body_state[
            :, self.wrist_indices, :7
        ]  # [num_envs, 2, 7], two hands
        wrist_pos = wrist_pos[:, :, :3]  # [num_envs, 2, 3], two hands, position only
        box_pos = envstates.objects["box"].root_state[:, :3]
        diff = box_pos - self.box_goal_pos
        wrist_box_diff = wrist_pos - box_pos.unsqueeze(1)

        wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 6]
        wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1)  # [num_envs, 6]
        box_goal_pos_obs = torch.flatten(self.box_goal_pos, start_dim=1)  # [num_envs, 3]
        box_pos_obs = torch.flatten(box_pos, start_dim=1)  # [num_envs, 3]
        diff_obs = torch.flatten(diff, start_dim=1)  # [num_envs, 3]

        self.privileged_obs_buf = torch.cat(
            (
                box_goal_pos_obs,  # 3
                box_pos_obs,  # 3
                diff_obs,
                wrist_pos_obs,
                wrist_box_diff_obs,
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
                diff_obs,  # 3
                wrist_box_diff_obs,  # 6
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            ),
            dim=-1,
        )
        # TODO noise implementation as original repo
        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

    def _resample_box_goal(self, env_ids, env_states):
        self.box_goal_pos[env_ids, 0] = env_states.objects["box"].root_state[env_ids, :0]
        self.box_goal_pos[env_ids, 1] = env_states.objects["box"].root_state[env_ids, :1]
        self.box_goal_pos[env_ids, 2] = env_states.objects["box"].root_state[env_ids, 2] + torch.FloatTensor(
            len(env_ids)
        ).uniform_(*self.cfg.command_ranges.box_pos_z).to(self.device)
