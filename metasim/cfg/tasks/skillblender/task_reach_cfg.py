"""FarReaching config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
from metasim.types import EnvState
from metasim.utils import configclass


# define new reward function
def reward_wrist_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    wrist_pos = env_states.robots[robot_name].body_state[:, cfg.wrist_indices, :7]  # [num_envs, 2, 7], two hands
    wrist_pos_diff = (
        wrist_pos[:, :, :3] - env_states.robots[robot_name].extra["ref_wrist_pos"][:, :, :3]
    )  # [num_envs, 2, 3], two hands, position only
    wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [num_envs, 6]
    wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_pos_error), wrist_pos_error


@configclass
class TaskReachCfgPPO(BaseHumanoidCfgPPO):
    seed = 5

    @configclass
    class Policy(BaseHumanoidCfgPPO.Policy):
        """Network config class for PPO."""

        num_dofs = 19
        frame_stack = 1
        command_dim = 14
        skill_dict = {
            "h1_wrist_walking": {
                "experiment_name": "h1_wrist_walking",
                "load_run": "2025_0101_093233",
                "checkpoint": -1,
                "low_high": (-2, 2),
            },
            "h1_wrist_reaching": {
                "experiment_name": "h1_wrist_reaching",
                "load_run": "2025_0621_134216",
                "checkpoint": -1,
                "low_high": (-1, 1),
            },
        }

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        policy_class_name = "ActorCriticHierarchical"
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # 3001  # number of policy updates
        save_interval = 400
        experiment_name = "task_reach"

    policy = Policy()
    runner = Runner()


@configclass
class TaskReachCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:FarReaching."""

    task_name = "task_reach"
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

    ppo_cfg = TaskReachCfgPPO()
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
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    reward_functions: list[Callable] = [
        reward_wrist_pos,
    ]

    reward_weights: dict[str, float] = {
        "wrist_pos": 5,
    }

    def __post_init__(self):
        super().__post_init__()
        self.command_ranges.wrist_max_radius = 0.25
        self.command_ranges.l_wrist_pos_x = [-0.10, 0.25]
        self.command_ranges.l_wrist_pos_y = [-0.10, 0.25]
        self.command_ranges.l_wrist_pos_z = [-0.25, 0.25]
        self.command_ranges.r_wrist_pos_x = [-0.10, 0.25]
        self.command_ranges.r_wrist_pos_y = [-0.25, 0.10]
        self.command_ranges.r_wrist_pos_z = [-0.25, 0.25]
