"""Reaching config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_default_joint_pos,
    reward_dof_acc,
    reward_dof_vel,
    reward_feet_distance,
    reward_orientation,
    reward_torques,
    reward_upper_body_pos,
)
from metasim.types import EnvState
from metasim.utils import configclass


def reward_wrist_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    """Reward for reaching the target position"""
    wrist_pos = env_states.robots[robot_name].body_state[:, cfg.wrist_indices, :7]  # [num_envs, 2, 7], two hands
    wrist_pos_diff = (
        wrist_pos[:, :, :3] - env_states.robots[robot_name].extra["ref_wrist_pos"][:, :, :3]
    )  # [num_envs, 2, 3], two hands, position only
    wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [num_envs, 6]
    wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_pos_error), wrist_pos_error


@configclass
class ReachingCfgPPO(BaseHumanoidCfgPPO):
    seed = 5

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        algorithm_class_name = "PPO"
        max_iterations = 15001
        save_interval = 500
        experiment_name = "reaching"

    runner = Runner()


@configclass
class ReachingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Reaching."""

    task_name = "reaching"
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        substeps=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=False,
        friction_offset_threshold=0.04,
        num_threads=10,
    )

    ppo_cfg = ReachingCfgPPO()

    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )

    num_actions = 19
    frame_stack = 1
    c_frame_stack = 3
    command_dim = 14
    num_single_obs = 3 * num_actions + 6 + command_dim
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 60

    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    reward_functions: list[Callable] = [
        reward_wrist_pos,
        reward_upper_body_pos,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_feet_distance,
        reward_default_joint_pos,
    ]

    reward_weights: dict[str, float] = {
        "wrist_pos": 5,
        "feet_distance": 0.5,
        "upper_body_pos": 0.5,
        "default_joint_pos": 0.5,
        "orientation": 1.0,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 3 * self.num_actions + 60
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
        self.command_ranges.wrist_max_radius = 0.25
        self.command_ranges.l_wrist_pos_x = [-0.10, 0.25]
        self.command_ranges.l_wrist_pos_y = [-0.10, 0.25]
        self.command_ranges.l_wrist_pos_z = [-0.25, 0.25]
        self.command_ranges.r_wrist_pos_x = [-0.10, 0.25]
        self.command_ranges.r_wrist_pos_y = [-0.25, 0.10]
        self.command_ranges.r_wrist_pos_z = [-0.25, 0.25]
