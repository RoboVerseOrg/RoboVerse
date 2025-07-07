from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class G1Cfg(BaseRobotCfg):
    name: str = "g1"
    num_joints: int = 21
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_lock_waist_rev_1_0_modified.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll": BaseActuatorCfg(stiffness=150, damping=5),
        "left_hip_yaw": BaseActuatorCfg(stiffness=150, damping=5),
        "left_knee": BaseActuatorCfg(stiffness=200, damping=5),
        "left_ankle_pitch": BaseActuatorCfg(stiffness=20, damping=4),
        "left_ankle_roll": BaseActuatorCfg(stiffness=20, damping=4),
        "right_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll": BaseActuatorCfg(stiffness=150, damping=5),
        "right_hip_yaw": BaseActuatorCfg(stiffness=150, damping=5),
        "right_knee": BaseActuatorCfg(stiffness=200, damping=5),
        "right_ankle_pitch": BaseActuatorCfg(stiffness=20, damping=4),
        "right_ankle_roll": BaseActuatorCfg(stiffness=20, damping=4),
        "waist_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "left_shoulder_pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow": BaseActuatorCfg(stiffness=40, damping=10),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_pitch": (-2.5307, 2.8798),
        "left_hip_roll": (-0.5236, 2.9671),
        "left_hip_yaw": (-2.7576, 2.7576),
        "left_knee": (-0.087267, 2.8798),
        "left_ankle_pitch": (-0.87267, 0.5236),
        "left_ankle_roll": (-0.2618, 0.2618),
        "right_hip_pitch": (-2.5307, 2.8798),
        "right_hip_roll": (-2.9671, 0.5236),
        "right_hip_yaw": (-2.7576, 2.7576),
        "right_knee": (-0.087267, 2.8798),
        "right_ankle_pitch": (-0.87267, 0.5236),
        "right_ankle_roll": (-0.2618, 0.2618),
        "waist_yaw": (-2.618, 2.618),
        "left_shoulder_pitch": (-3.0892, 2.6704),
        "left_shoulder_roll": (-1.5882, 2.2515),
        "left_shoulder_yaw": (-2.618, 2.618),
        "left_elbow": (-1.0472, 2.0944),
        "right_shoulder_pitch": (-3.0892, 2.6704),
        "right_shoulder_roll": (-2.2515, 1.5882),
        "right_shoulder_yaw": (-2.618, 2.618),
        "right_elbow": (-1.0472, 2.0944),
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "left_hip_pitch": -0.4,
        "left_hip_roll": 0,
        "left_hip_yaw": 0.0,
        "left_knee": 0.8,
        "left_ankle_pitch": -0.4,
        "left_ankle_roll": 0,
        "right_hip_pitch": -0.4,
        "right_hip_roll": 0,
        "right_hip_yaw": 0.0,
        "right_knee": 0.8,
        "right_ankle_pitch": -0.4,
        "right_ankle_roll": 0,
        "waist_yaw": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 0.0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "left_hip_pitch": "effort",
        "left_hip_roll": "effort",
        "left_hip_yaw": "effort",
        "left_knee": "effort",
        "left_ankle_pitch": "effort",
        "left_ankle_roll": "effort",
        "right_hip_pitch": "effort",
        "right_hip_roll": "effort",
        "right_hip_yaw": "effort",
        "right_knee": "effort",
        "right_ankle_pitch": "effort",
        "right_ankle_roll": "effort",
        "waist_yaw": "effort",
        "left_shoulder_pitch": "effort",
        "left_shoulder_roll": "effort",
        "left_shoulder_yaw": "effort",
        "left_elbow": "effort",
        "right_shoulder_pitch": "effort",
        "right_shoulder_roll": "effort",
        "right_shoulder_yaw": "effort",
        "right_elbow": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = [
        "ankle_roll",
    ]
    knee_links: list[str] = [
        "knee",
    ]
    elbow_links: list[str] = [
        "elbow",
    ]
    torso_links: list[str] = ["torso_link"]

    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]

    penalized_contacts_links: list[str] = ["hip", "knee"]
