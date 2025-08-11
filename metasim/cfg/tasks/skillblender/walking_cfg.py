"""Walking config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
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


@configclass
class WalkingCfgPPO(BaseHumanoidCfgPPO):
    seed: int = 0

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        num_steps_per_env = 60
        max_iterations = 15001
        save_interval = 500
        experiment_name = "walking"

    runner: Runner = Runner()


@configclass
class WalkingCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:Walking."""

    task_name = "walking"

    ppo_cfg = WalkingCfgPPO()

    command_dim = 3
    frame_stack = 1
    c_frame_stack = 3
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

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
        "joint_pos": 1.6,
        "feet_clearance": 2.0,
        "feet_contact_number": 2.4,
        # gait
        "foot_slip": -0.05,
        "feet_distance": 0.2,
        "knee_distance": 0.2,
        # contact
        "feet_contact_forces": -0.01,
        # vel tracking
        "tracking_lin_vel": 2.4,
        "tracking_ang_vel": 2.2,
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.2,
        "track_vel_hard": 1.0,
        # base pos
        "default_joint_pos": 1.0,
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

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 4 * self.num_actions + 25
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
