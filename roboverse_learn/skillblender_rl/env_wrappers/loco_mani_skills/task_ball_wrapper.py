"""SkillBlench wrapper for training loco-manipulation Skillbench:FootballShoot"""

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


class TaskBallWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:FootballShoot
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        _, _ = self.env.reset(self.init_states)
        self.env.handler.simulate()

    def _init_buffers(self):
        super()._init_buffers()
        self.ori_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # TODO add domain randomizatoin
        self.ori_ball_pos[:, 0] = 0.5 * (self.scenario.task.ball_range_x[0] + self.scenario.task.ball_range_x[1])
        self.ori_ball_pos[:, 1] = 0.5 * (self.scenario.task.ball_range_y[0] + self.scenario.task.ball_range_y[1])
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # HACK This is a hack to get the goal position for checker
        self.cfg.goal_pos = self.goal_pos

    def _parse_goal_pos(self, envstate: EnvState):
        """Parse goal position from envstate"""
        envstate.robots[self.robot.name].extra["goal_pos"] = self.goal_pos
        envstate.robots[self.robot.name].extra["ori_ball_pos"] = self.ori_ball_pos

    def _parse_state_for_reward(self, envstate: EnvState) -> None:
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """
        # TODO read from config
        # parse those state which cannot directly get from Envstates
        super()._parse_state_for_reward(envstate)
        self._parse_goal_pos(envstate)

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

        # ball observations
        ball_pos = envstates.objects[self.scenario.objects[0].name].root_state[:, :3]
        torso_pos = envstates.robots[self.robot.name].body_state[:, self.torso_indices, :3].squeeze(1)
        ball_goal_diff = ball_pos - self.goal_pos
        root_ball_diff = torso_pos - ball_pos

        goal_pos_obs = torch.flatten(self.goal_pos, start_dim=1)  # [num_envs, 3]
        ball_pos_obs = torch.flatten(ball_pos, start_dim=1)  # [num_envs, 3]
        torso_pos_obs = torch.flatten(torso_pos, start_dim=1)  # [num_envs, 3]
        ball_goal_diff_obs = torch.flatten(ball_goal_diff, start_dim=1)  # [num_envs, 3]
        root_ball_diff_obs = torch.flatten(root_ball_diff, start_dim=1)  # [num_envs, 3]

        self.privileged_obs_buf = torch.cat(
            (
                goal_pos_obs,  # 3
                ball_pos_obs,  # 3
                torso_pos_obs,  # 3
                ball_goal_diff_obs,  # 3
                root_ball_diff_obs,  # 3
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
                ball_goal_diff_obs,  # 3
                root_ball_diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
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
