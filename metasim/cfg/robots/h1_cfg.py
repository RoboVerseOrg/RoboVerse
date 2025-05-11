from __future__ import annotations

from typing import Literal

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
        "left_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "left_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "left_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "right_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "right_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "right_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "torso": BaseActuatorCfg(stiffness=300, damping=6),
        "left_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "left_elbow": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "right_elbow": BaseActuatorCfg(stiffness=100, damping=2),
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

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
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
        "left_shoulder_roll": 0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "left_hip_yaw": "effort",
        "left_hip_roll": "effort",
        "left_hip_pitch": "effort",
        "left_knee": "effort",
        "left_ankle": "effort",
        "right_hip_yaw": "effort",
        "right_hip_roll": "effort",
        "right_hip_pitch": "effort",
        "right_knee": "effort",
        "right_ankle": "effort",
        "torso": "effort",
        "left_shoulder_pitch": "effort",
        "left_shoulder_roll": "effort",
        "left_shoulder_yaw": "effort",
        "left_elbow": "effort",
        "right_shoulder_pitch": "effort",
        "right_shoulder_roll": "effort",
        "right_shoulder_yaw": "effort",
        "right_elbow": "effort",
    }
