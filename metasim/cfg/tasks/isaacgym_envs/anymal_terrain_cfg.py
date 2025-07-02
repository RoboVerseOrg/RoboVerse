from __future__ import annotations

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.anymal_cfg import AnymalCfg as AnymalRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AnymalTerrainCfg(BaseTaskCfg):
    name = "isaacgym_envs:AnymalTerrain"
    episode_length = 1000
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    terrain_type = "plane"
    terrain_curriculum = True
    terrain_num_levels = 5
    terrain_num_terrains = 8
    terrain_map_length = 8.0
    terrain_map_width = 8.0
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    terrain_static_friction = 1.0
    terrain_dynamic_friction = 1.0
    terrain_restitution = 0.0
    terrain_slope_threshold = 0.75

    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    height_meas_scale = 5.0
    action_scale = 0.5

    terminal_reward = -0.0
    lin_vel_xy_reward_scale = 1.0
    lin_vel_z_reward_scale = -4.0
    ang_vel_z_reward_scale = 0.5
    ang_vel_xy_reward_scale = -0.05
    orient_reward_scale = -0.0
    torque_reward_scale = -0.00001
    joint_acc_reward_scale = -0.0005
    base_height_reward_scale = -0.0
    feet_air_time_reward_scale = 1.0
    knee_collision_reward_scale = -0.25
    feet_stumble_reward_scale = -0.0
    action_rate_reward_scale = -0.01
    hip_reward_scale = -0.0

    command_x_range = [-1.0, 1.0]
    command_y_range = [-1.0, 1.0]
    command_yaw_range = [-1.0, 1.0]

    base_init_state = [0.0, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    default_joint_angles = {
        "LF_HAA": 0.03,
        "LH_HAA": 0.03,
        "RF_HAA": -0.03,
        "RH_HAA": -0.03,
        "LF_HFE": 0.4,
        "LH_HFE": -0.4,
        "RF_HFE": 0.4,
        "RH_HFE": -0.4,
        "LF_KFE": -0.8,
        "LH_KFE": 0.8,
        "RF_KFE": -0.8,
        "RH_KFE": 0.8,
    }

    decimation = 4
    control_frequency_inv = 1
    push_interval_s = 15.0
    allow_knee_contacts = False

    kp = 50.0
    kd = 2.0

    add_noise = True
    noise_level = 1.0
    linear_velocity_noise = 1.5
    angular_velocity_noise = 0.2
    gravity_noise = 0.05
    dof_position_noise = 0.01
    dof_velocity_noise = 1.5
    height_measurement_noise = 0.1

    friction_range = [0.5, 1.25]

    robot: AnymalRobotCfg = AnymalRobotCfg()

    objects: list[RigidObjCfg] = []

    control: ControlCfg = ControlCfg(action_scale=0.5, action_offset=True, torque_limit_scale=1.0)

    checker = EmptyChecker()

    observation_space = {"shape": [188]}
