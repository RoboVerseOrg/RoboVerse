from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1Dof12Cfg(RobotCfg):
    name: str = "g1_dof12"
    num_joints: int = 12
    usd_path: str = MISSING
    xml_path: str = "roboverse_data/robots/g1/xml/g1_12dof.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_12dof.urdf"
    mjcf_path = xml_path
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=139),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88),
        "left_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, torque_limit=139),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=35),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=35),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=139),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88),
        "right_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, torque_limit=139),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=35),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=35),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Hips & legs
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
    }

    torque_limits: dict[str, float] = {
        # Hips & legs
        "left_hip_pitch_joint": 88,
        "left_hip_roll_joint": 139,
        "left_hip_yaw_joint": 88,
        "left_knee_joint": 139,
        "left_ankle_pitch_joint": 35,
        "left_ankle_roll_joint": 35,
        "right_hip_pitch_joint": 88,
        "right_hip_roll_joint": 139,
        "right_hip_yaw_joint": 88,
        "right_knee_joint": 139,
        "right_ankle_pitch_joint": 35,
        "right_ankle_roll_joint": 35,
    }

    default_joint_positions: dict[str, float] = {
        # Hips & legs
        "left_hip_pitch_joint": -0.1,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # Hips & legs
        "left_hip_pitch_joint": "effort",
        "left_hip_roll_joint": "effort",
        "left_hip_yaw_joint": "effort",
        "left_knee_joint": "effort",
        "left_ankle_pitch_joint": "effort",
        "left_ankle_roll_joint": "effort",
        "right_hip_pitch_joint": "effort",
        "right_hip_roll_joint": "effort",
        "right_hip_yaw_joint": "effort",
        "right_knee_joint": "effort",
        "right_ankle_pitch_joint": "effort",
        "right_ankle_roll_joint": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = []
    wrist_links: list[str] = []
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    # joint substrings, to find indices of joints.
    left_yaw_roll_joints = ["left_hip_yaw_joint", "left_hip_roll_joint"]
    right_yaw_roll_joints = ["right_hip_yaw_joint", "right_hip_roll_joint"]
    upper_body_joints = []

    # From default joint armature in XML
    dof_armature: float = 0.0
