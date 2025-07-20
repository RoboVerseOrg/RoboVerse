from __future__ import annotations

from metasim.cfg.robots import BaseActuatorCfg, BaseRobotCfg
from metasim.utils import configclass


@configclass
class H1VerseCfg(BaseRobotCfg):
    name: str = "h1_verse"
    num_joints: int = 19
    urdf_path: str = "roboverse_data/robots/h1_verse/urdf/h1.urdf"
    enabled_gravity = True
    fix_base_link = False
    collapse_fixed_joints = True
    decimation: int = 4

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_yaw":        BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll":       BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_pitch":      BaseActuatorCfg(stiffness=200, damping=5),
        "left_knee":           BaseActuatorCfg(stiffness=300, damping=6),
        "left_ankle":          BaseActuatorCfg(stiffness=40,  damping=2),

        "right_hip_yaw":       BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll":      BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_pitch":     BaseActuatorCfg(stiffness=200, damping=5),
        "right_knee":          BaseActuatorCfg(stiffness=300, damping=6),
        "right_ankle":         BaseActuatorCfg(stiffness=40,  damping=2),

        "torso":               BaseActuatorCfg(stiffness=300, damping=6),

        "left_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_roll":  BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_yaw":   BaseActuatorCfg(stiffness=100, damping=2),
        "left_elbow":          BaseActuatorCfg(stiffness=100, damping=2),

        "right_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_roll":  BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_yaw":   BaseActuatorCfg(stiffness=100, damping=2),
        "right_elbow":          BaseActuatorCfg(stiffness=100, damping=2),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_yaw": (-0.43, 0.43),  "right_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43), "right_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-1.57, 1.57), "right_hip_pitch": (-1.57, 1.57),
        "left_knee": (-0.26, 2.05),     "right_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),    "right_ankle": (-0.87, 0.52),

        "torso": (-2.35, 2.35),

        "left_shoulder_pitch": (-2.87, 2.87), "right_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll":  (-0.34, 3.11), "right_shoulder_roll": (-3.11, 0.34),
        "left_shoulder_yaw":   (-1.30, 4.45), "right_shoulder_yaw": (-4.45, 1.30),
        "left_elbow": (-1.25, 2.61),          "right_elbow": (-1.25, 2.61),
    }
