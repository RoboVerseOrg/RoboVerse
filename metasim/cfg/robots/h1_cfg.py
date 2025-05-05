from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class H1Cfg(BaseRobotCfg):
    name: str = "h1"
    num_joints: int = 26
    usd_path: str = "roboverse_data/robots/h1/usd/h1.usd"
    mjcf_path: str = "roboverse_data/robots/h1/mjcf/h1.xml"
    urdf_path: str = "roboverse_data/robots/h1/urdf/h1.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_yaw": BaseActuatorCfg(),
        "left_hip_roll": BaseActuatorCfg(),
        "left_hip_pitch": BaseActuatorCfg(),
        "left_knee": BaseActuatorCfg(),
        "left_ankle": BaseActuatorCfg(),
        "right_hip_yaw": BaseActuatorCfg(),
        "right_hip_roll": BaseActuatorCfg(),
        "right_hip_pitch": BaseActuatorCfg(),
        "right_knee": BaseActuatorCfg(),
        "right_ankle": BaseActuatorCfg(),
        "torso": BaseActuatorCfg(),
        "left_shoulder_pitch": BaseActuatorCfg(),
        "left_shoulder_roll": BaseActuatorCfg(),
        "left_shoulder_yaw": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
        "right_shoulder_pitch": BaseActuatorCfg(),
        "right_shoulder_roll": BaseActuatorCfg(),
        "right_shoulder_yaw": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
    }
