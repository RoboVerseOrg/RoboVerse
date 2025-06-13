"""Walking config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg
from metasim.cfg.tasks.skillblender.base_legged_cfg import (
    BaseConfig,
    CommandRanges,
    CommandsConfig,
    LeggedRobotCfgPPO,
    RewardCfg,
)
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_dof_acc,
    reward_dof_vel,
    reward_orientation,
    reward_torques,
    reward_upper_body_pos,
)
from metasim.types import EnvState

# from metasim.cfg.tasks.skillblender.reward_func_cfg import *  # FIXME star import
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import *


# define new reward function
def reward_feet_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    foot_pos = env_states.robots[robot_name].body_state[:, cfg.feet_indices, :2]
    feet_pos_diff = (
        foot_pos[:, :, :2] - env_states.robots[robot_name].extra["ref_feet_pos"][:, :, :2]
    )  # [num_envs, 2, 2], two feet, position only
    feet_pos_diff = torch.flatten(feet_pos_diff, start_dim=1)  # [num_envs, 4]
    feet_pos_error = torch.mean(torch.abs(feet_pos_diff), dim=1)
    return torch.exp(-4 * feet_pos_error), feet_pos_error


# ppo config
class SteppingCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        wandb = True
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # 3001  # number of policy updates

        # logging
        save_interval = 5000  # check for potential saves every this many iterations
        experiment_name = "h1_stepping"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt


# TODO task config override robot config
class robot_asset(BaseConfig):
    fix_base_link: bool = False
    penalize_contacts_on = ["hip", "knee", "pelvis", "torso", "shoulder", "elbow"]


@configclass
class SteppingRewardCfg(RewardCfg):
    base_height_target = 0.89
    min_dist = 0.2
    max_dist = 0.5
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale = 0.17  # rad
    target_feet_height = 0.06  # m
    cycle_time = 0.64  # sec
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    # tracking reward = exp(error*sigma)
    tracking_sigma = 5
    max_contact_force = 700  # forces above this value are penalized


@configclass
class SteppingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Stepping."""

    task_name = "walking"
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        substeps=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=True,
        friction_offset_threshold=0.04,
        num_threads=10,
    )

    ppo_cfg = SteppingCfgPPO()
    reward_cfg = SteppingRewardCfg()
    command_ranges = CommandRanges(lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0])
    command_ranges.feet_max_radius = 0.25

    num_actions = 19
    command_dim = 4
    frame_stack = 1
    c_frame_stack = 3
    num_single_obs = 3 * num_actions + 6 + command_dim  #
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 18 + 12
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    commands = CommandsConfig(num_commands=4, resampling_time=8.0)
    traj_filepath = "roboverse_data/trajs/humanoidbench/stand/v2/initial_state_v2.json"

    reward_functions: list[Callable] = [
        reward_feet_pos,
        reward_upper_body_pos,
        reward_orientation,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
    ]

    # TODO: check why this configuration not work as well as the original one, that is probably a bug in infra.

    reward_weights: dict[str, float] = {
        "feet_pos": 5,
        "upper_body_pos": 0.5,
        "orientation": 1.0,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
    }
