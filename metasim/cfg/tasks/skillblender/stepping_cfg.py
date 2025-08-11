"""Stepping config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_dof_acc,
    reward_dof_vel,
    reward_orientation,
    reward_torques,
    reward_upper_body_pos,
)
from metasim.types import EnvState
from metasim.utils import configclass


def reward_feet_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    """Reward function for feet position."""
    foot_pos = env_states.robots[robot_name].body_state[:, cfg.feet_indices, :2]
    feet_pos_diff = (
        foot_pos[:, :, :2] - env_states.robots[robot_name].extra["ref_feet_pos"][:, :, :2]
    )  # [num_envs, 2, 2], two feet, position only
    feet_pos_diff = torch.flatten(feet_pos_diff, start_dim=1)  # [num_envs, 4]
    feet_pos_error = torch.mean(torch.abs(feet_pos_diff), dim=1)
    return torch.exp(-4 * feet_pos_error), feet_pos_error


@configclass
class SteppingCfgPPO(BaseHumanoidCfgPPO):
    """PPO config for Skillbench:Stepping."""

    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        """Runner config for Skillbench:Stepping."""

        wandb = True
        num_steps_per_env = 60
        max_iterations = 15001
        save_interval = 500
        experiment_name = "stepping"

    runner = Runner()


@configclass
class SteppingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Stepping."""

    task_name = "stepping"

    ppo_cfg = SteppingCfgPPO()

    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    command_dim = 4
    frame_stack = 1
    c_frame_stack = 3

    reward_functions: list[Callable] = [
        reward_feet_pos,
        reward_upper_body_pos,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
    ]
    reward_weights: dict[str, float] = {
        "feet_pos": 5,
        "upper_body_pos": 0.5,
        "orientation": 1.0,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 3 * self.num_actions + 18 + 12
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
        self.command_ranges.feet_max_radius = 0.25
