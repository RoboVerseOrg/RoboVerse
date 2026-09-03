from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1Dof12Cfg(RobotCfg):  # noqa: D101
    name: str = "g1_dof12"
    num_joints: int = 12
    usd_path: str = "roboverse_data/robots/g1/usd/g1_12dof.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_12dof.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_12dof.urdf"
    mjcf_path = xml_path
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = True
    isaacgym_read_mjcf = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=88, velocity_limit=32.0),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=88, velocity_limit=32.0),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=88, velocity_limit=32.0),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=88, velocity_limit=32.0),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=139, velocity_limit=20.0),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, effort_limit_sim=139, velocity_limit=20.0),
        "left_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, effort_limit_sim=139, velocity_limit=20.0),
        "right_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, effort_limit_sim=139, velocity_limit=20.0),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, effort_limit_sim=25, velocity_limit=37.0),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, effort_limit_sim=25, velocity_limit=37.0),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, effort_limit_sim=25, velocity_limit=37.0),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, effort_limit_sim=25, velocity_limit=37.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
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

    default_joint_positions: dict[str, float] = {
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

    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    torso_links: list[str] = ["torso_link"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["rubber_hand"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee", "shoulder", "elbow", "wrist"]

    left_hip_yaw_roll_joints: list[str] = ["left_hip_yaw_joint", "left_hip_roll_joint"]
    right_hip_yaw_roll_joints: list[str] = ["right_hip_yaw_joint", "right_hip_roll_joint"]

    soft_joint_pos_limit_factor = 0.9
    armature: float = 0.01


@configclass
class G1Dof23Cfg(G1Dof12Cfg):  # noqa: D101
    name: str = "g1_dof23"
    num_joints: int = 23
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = MISSING
    mjcf_path = xml_path

    actuators = {
        **G1Dof12Cfg().actuators,
        "waist_yaw_joint": BaseActuatorCfg(stiffness=200, damping=5, effort_limit_sim=88, velocity_limit=32.0),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "right_shoulder_pitch_joint": BaseActuatorCfg(
            stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0
        ),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=25, velocity_limit=37.0),
    }

    joint_limits = {
        **G1Dof12Cfg().joint_limits,
        "waist_yaw_joint": (-2.618, 2.618),
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222, 1.972222),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_roll_joint": (-1.972222, 1.972222),
    }

    default_joint_positions = {
        **G1Dof12Cfg().default_joint_positions,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
    }

    control_type = {
        **G1Dof12Cfg().control_type,
        "waist_yaw_joint": "effort",
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "left_wrist_roll_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        "right_wrist_roll_joint": "effort",
    }


@configclass
class G1Dof27Cfg(G1Dof23Cfg):  # noqa: D101
    name: str = "g1_dof27"
    num_joints: int = 27
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = MISSING
    mjcf_path = xml_path

    actuators = {
        **G1Dof23Cfg().actuators,
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=5, velocity_limit=22.0),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=5, velocity_limit=22.0),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=5, velocity_limit=22.0),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, effort_limit_sim=5, velocity_limit=22.0),
    }

    joint_limits = {
        **G1Dof23Cfg().joint_limits,
        "left_wrist_pitch_joint": (-1.61443, 1.61443),
        "left_wrist_yaw_joint": (-1.61443, 1.61443),
        "right_wrist_pitch_joint": (-1.61443, 1.61443),
        "right_wrist_yaw_joint": (-1.61443, 1.61443),
    }

    default_joint_positions = {
        **G1Dof23Cfg().default_joint_positions,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    control_type = {
        **G1Dof23Cfg().control_type,
        "left_wrist_pitch_joint": "effort",
        "left_wrist_yaw_joint": "effort",
        "right_wrist_pitch_joint": "effort",
        "right_wrist_yaw_joint": "effort",
    }


@configclass
class G1Dof29Cfg(G1Dof27Cfg):  # noqa: D101
    name: str = "g1_dof29"
    num_joints: int = 29
    usd_path: str = "roboverse_data/robots/g1/usd/g1_29dof_rev_1_0.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_29dof_rev_1_0.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_rev_1_0.urdf"
    mjcf_path = xml_path

    actuators = {
        **G1Dof27Cfg().actuators,
        "waist_roll_joint": BaseActuatorCfg(stiffness=40, damping=5, effort_limit_sim=25, velocity_limit=37.0),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=5, effort_limit_sim=25, velocity_limit=37.0),
    }

    joint_limits = {
        **G1Dof27Cfg().joint_limits,
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
    }

    default_joint_positions = {
        **G1Dof27Cfg().default_joint_positions,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
    }

    control_type = {
        **G1Dof27Cfg().control_type,
        "waist_roll_joint": "effort",
        "waist_pitch_joint": "effort",
    }
