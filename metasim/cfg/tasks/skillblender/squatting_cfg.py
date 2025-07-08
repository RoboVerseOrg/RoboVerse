"""Squattting config in SkillBench in Skillblender"""

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


# define new reward function
def reward_squatting(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    """
    Calculates the reward based on the difference between the current root height and the target root height.
    """
    root_height = env_states.robots[robot_name].root_state[:, 2].unsqueeze(1)
    ref_root_height = env_states.robots[robot_name].extra["ref_root_height"]
    root_height_diff = root_height - ref_root_height  # [num_envs, 1]
    root_height_error = torch.mean(torch.abs(root_height_diff), dim=1)
    return torch.exp(-4 * root_height_error), root_height_error


@configclass
class SquattingCfgPPO(BaseHumanoidCfgPPO):
    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # 3001  # number of policy updates
        save_interval = 500
        experiment_name = "squatting"

    runner: Runner = Runner()


@configclass
class SquattingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Squatting."""

    task_name = "squatting"
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

    ppo_cfg = SquattingCfgPPO()
    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )

    command_dim = 1
    c_frame_stack = 3
    frame_stack = 1

    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=10.0)

    reward_functions: list[Callable] = [
        reward_squatting,
        reward_upper_body_pos,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_feet_distance,
        reward_default_joint_pos,
    ]

    reward_weights: dict[str, float] = {
        "squatting": 5,
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
        self.single_num_privileged_obs: int = 3 * self.num_actions + 18 + 3
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)

        self.command_ranges.root_height_std = 0.2
        self.command_ranges.min_root_height = 0.2
        self.command_ranges.max_root_height = 1.1
