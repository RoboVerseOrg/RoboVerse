from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class H12WithoutHandCfg(BaseRobotCfg):
    name: str = "h1_2_without_hand"
    num_joints: int = 27
    usd_path: str = "roboverse_data_release/robots/h1_2_without_hand/usd/h1_2_without_hand.usd"
    enabled_gravity: bool = True  # ! critical
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "torso_joint": BaseActuatorCfg(),
        "left_shoulder_pitch_joint": BaseActuatorCfg(),
        "right_shoulder_pitch_joint": BaseActuatorCfg(),
        "left_hip_yaw_joint": BaseActuatorCfg(),
        "right_hip_yaw_joint": BaseActuatorCfg(),
        "left_shoulder_roll_joint": BaseActuatorCfg(),
        "right_shoulder_roll_joint": BaseActuatorCfg(),
        "left_hip_pitch_joint": BaseActuatorCfg(),
        "right_hip_pitch_joint": BaseActuatorCfg(),
        "left_shoulder_yaw_joint": BaseActuatorCfg(),
        "right_shoulder_yaw_joint": BaseActuatorCfg(),
        "left_hip_roll_joint": BaseActuatorCfg(),
        "right_hip_roll_joint": BaseActuatorCfg(),
        "left_elbow_pitch_joint": BaseActuatorCfg(),
        "right_elbow_pitch_joint": BaseActuatorCfg(),
        "left_knee_joint": BaseActuatorCfg(),
        "right_knee_joint": BaseActuatorCfg(),
        "left_elbow_roll_joint": BaseActuatorCfg(),
        "right_elbow_roll_joint": BaseActuatorCfg(),
        "left_ankle_pitch_joint": BaseActuatorCfg(),
        "right_ankle_pitch_joint": BaseActuatorCfg(),
        "left_wrist_pitch_joint": BaseActuatorCfg(),
        "right_wrist_pitch_joint": BaseActuatorCfg(),
        "left_ankle_roll_joint": BaseActuatorCfg(),
        "right_ankle_roll_joint": BaseActuatorCfg(),
        "left_wrist_yaw_joint": BaseActuatorCfg(),
        "right_wrist_yaw_joint": BaseActuatorCfg(),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "torso_joint": (-2.3500, 2.3500),
        "left_shoulder_pitch_joint": (-3.1400, 1.5700),
        "right_shoulder_pitch_joint": (-3.1400, 1.5700),
        "left_hip_yaw_joint": (-0.4300, 0.4300),
        "right_hip_yaw_joint": (-0.4300, 0.4300),
        "left_shoulder_roll_joint": (-0.3800, 3.4000),
        "right_shoulder_roll_joint": (-3.4000, 0.3800),
        "left_hip_pitch_joint": (-3.1400, 2.5000),
        "right_hip_pitch_joint": (-3.1400, 2.5000),
        "left_shoulder_yaw_joint": (-2.6600, 3.0100),
        "right_shoulder_yaw_joint": (-3.0100, 2.6600),
        "left_hip_roll_joint": (-0.4300, 3.1400),
        "right_hip_roll_joint": (-3.1400, 0.4300),
        "left_elbow_pitch_joint": (-0.9500, 3.1800),
        "right_elbow_pitch_joint": (-0.9500, 3.1800),
        "left_knee_joint": (-0.2600, 2.0500),
        "right_knee_joint": (-0.2600, 2.0500),
        "left_elbow_roll_joint": (-3.0100, 2.7500),
        "right_elbow_roll_joint": (-2.7500, 3.0100),
        "left_ankle_pitch_joint": (-0.8973, 0.5236),
        "right_ankle_pitch_joint": (-0.8973, 0.5236),
        "left_wrist_pitch_joint": (-0.4700, 0.4700),
        "right_wrist_pitch_joint": (-0.4700, 0.4700),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
        "left_wrist_yaw_joint": (-1.2700, 1.2700),
        "right_wrist_yaw_joint": (-1.2700, 1.2700),
    }
