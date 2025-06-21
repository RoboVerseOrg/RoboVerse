"""Walking config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg
from metasim.cfg.tasks.skillblender.base_legged_cfg import CommandRanges, CommandsConfig, LeggedRobotCfgPPO, RewardCfg
from metasim.cfg.tasks.skillblender.reward_func_cfg import (
    reward_action_rate,
    reward_action_smoothness,
    reward_ang_vel_xy,
    reward_base_acc,
    reward_base_height,
    reward_collision,
    reward_default_joint_pos,
    reward_dof_acc,
    reward_dof_pos_limits,
    reward_dof_vel,
    reward_dof_vel_limits,
    reward_elbow_distance,
    reward_feet_air_time,
    reward_feet_clearance,
    reward_feet_contact_forces,
    reward_feet_contact_number,
    reward_feet_distance,
    reward_foot_slip,
    reward_joint_pos,
    reward_knee_distance,
    reward_lin_vel_z,
    reward_low_speed,
    reward_orientation,
    reward_stand_still,
    reward_stumble,
    reward_termination,
    reward_torque_limits,
    reward_torques,
    reward_track_vel_hard,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
    reward_upper_body_pos,
    reward_vel_mismatch_exp,
)
from metasim.utils import configclass


class WalkingCfgPPO(LeggedRobotCfgPPO):
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
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "walking"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt


@configclass
class WalkingRewardCfg(RewardCfg):
    base_height_target = 0.89
    min_dist = 0.2
    max_dist = 0.5
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale = 0.17
    target_feet_height = 0.06
    cycle_time = 0.64
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    # tracking reward = exp(error*sigma)
    tracking_sigma = 5
    max_contact_force = 700
    soft_torque_limit = 0.001


@configclass
class WalkingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Walking."""

    task_name = "walking"
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

    ppo_cfg = WalkingCfgPPO()
    reward_cfg = WalkingRewardCfg()
    command_ranges = CommandRanges()

    num_actions = 19
    command_dim = 3
    frame_stack = 1
    c_frame_stack = 3
    num_single_obs = 3 * num_actions + 6 + command_dim  #
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 4 * num_actions + 25
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    commands = CommandsConfig(num_commands=4, resampling_time=8.0)

    reward_functions: list[Callable] = [
        reward_lin_vel_z,
        reward_ang_vel_xy,
        reward_orientation,
        reward_base_height,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_action_rate,
        reward_collision,
        reward_termination,
        reward_dof_pos_limits,
        reward_dof_vel_limits,
        reward_torque_limits,
        reward_tracking_lin_vel,
        reward_tracking_ang_vel,
        reward_feet_air_time,
        reward_stumble,
        reward_stand_still,
        reward_feet_contact_forces,
        reward_joint_pos,
        reward_feet_distance,
        reward_knee_distance,
        reward_elbow_distance,
        reward_foot_slip,
        reward_feet_contact_number,
        reward_default_joint_pos,
        reward_upper_body_pos,
        reward_base_acc,
        reward_vel_mismatch_exp,
        reward_track_vel_hard,
        reward_feet_clearance,
        reward_low_speed,
        reward_action_smoothness,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "lin_vel_z": -0.0,
        # "ang_vel_xy": -0.05,
        "base_height": 0.2,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "stand_still": -0.0,
        "joint_pos": 3.2,
        "feet_clearance": 2.0,
        "feet_contact_number": 2.4,
        # gait
        "foot_slip": -0.05,
        "feet_distance": 0.2,
        "knee_distance": 0.2,
        # contact
        "feet_contact_forces": -0.01,
        # vel tracking
        "tracking_lin_vel": 4.8,
        "tracking_ang_vel": 2.2,
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.2,
        "track_vel_hard": 1.0,
        # base pos
        "default_joint_pos": 0.5,
        "upper_body_pos": 0.5,
        "orientation": 1.0,
        "base_acc": 0.2,
        # energy
        "action_smoothness": -0.002,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
        "torque_limits": 0.001,
        # optional
        "action_rate": -0.0,
    }

    init_states = [
        {
            "objects": {},
            "robots": {
                "h1_wrist": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]
