from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class AgibotA2Dof12Cfg(RobotCfg):
    name: str = "agibot_a2_dof12"
    num_joints: int = 12
    urdf_path: str = "roboverse_data/robots/agibot_a2/urdf/agibot_a2_dof12.urdf"
    usd_path: str = "roboverse_data/robots/agibot_a2/usd/agibot_a2_dof12.usd"
    xml_path: str = "roboverse_data/robots/agibot_a2/mjcf/agibot_a2_dof12.mjcf"
    mjcf_path: str = xml_path

    isaacgym_flip_visual_attachments: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "idx01_left_hip_roll": BaseActuatorCfg(stiffness=130, damping=5.0, effort_limit_sim=76.8, velocity_limit=12.0),
        "idx02_left_hip_yaw": BaseActuatorCfg(stiffness=130, damping=3.0, effort_limit_sim=76.8, velocity_limit=12.0),
        "idx03_left_hip_pitch": BaseActuatorCfg(
            stiffness=200, damping=6.0, effort_limit_sim=216.0, velocity_limit=12.0
        ),
        "idx04_left_tarsus": BaseActuatorCfg(stiffness=220, damping=7.0, effort_limit_sim=216.0, velocity_limit=12.0),
        "idx05_left_toe_pitch": BaseActuatorCfg(stiffness=50, damping=1.5, effort_limit_sim=38.4, velocity_limit=12.0),
        "idx06_left_toe_roll": BaseActuatorCfg(stiffness=50, damping=1.5, effort_limit_sim=38.4, velocity_limit=12.0),
        "idx07_right_hip_roll": BaseActuatorCfg(stiffness=130, damping=5.0, effort_limit_sim=76.8, velocity_limit=12.0),
        "idx08_right_hip_yaw": BaseActuatorCfg(stiffness=130, damping=3.0, effort_limit_sim=76.8, velocity_limit=12.0),
        "idx09_right_hip_pitch": BaseActuatorCfg(
            stiffness=200, damping=6.0, effort_limit_sim=216.0, velocity_limit=12.0
        ),
        "idx10_right_tarsus": BaseActuatorCfg(stiffness=220, damping=7.0, effort_limit_sim=216.0, velocity_limit=12.0),
        "idx11_right_toe_pitch": BaseActuatorCfg(stiffness=50, damping=1.5, effort_limit_sim=38.4, velocity_limit=12.0),
        "idx12_right_toe_roll": BaseActuatorCfg(stiffness=50, damping=1.5, effort_limit_sim=38.4, velocity_limit=12.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Hips & legs
        "idx01_left_hip_roll": (-0.698, 0.698),
        "idx02_left_hip_yaw": (-1.57, 1.57),
        "idx03_left_hip_pitch": (-1.919862144, 0.78539815),
        "idx04_left_tarsus": (-0.087266461, 2.443460911),
        "idx05_left_toe_pitch": (-1.047197533, 0.523598767),
        "idx06_left_toe_roll": (-1.972222, 1.972222),
        "idx07_right_hip_roll": (-0.698, 0.698),
        "idx08_right_hip_yaw": (-1.57, 1.57),
        "idx09_right_hip_pitch": (-1.919862144, 0.78539815),
        "idx10_right_tarsus": (-0.087266461, 2.443460911),
        "idx11_right_toe_pitch": (-1.047197533, 0.523598767),
        "idx12_right_toe_roll": (-1.972222, 1.972222),
    }

    default_joint_positions: dict[str, float] = {
        # Hips & legs
        "idx01_left_hip_roll": 0.0,
        "idx02_left_hip_yaw": 0.0,
        "idx03_left_hip_pitch": -0.115,
        "idx04_left_tarsus": 0.267,
        "idx05_left_toe_pitch": -0.152,
        "idx06_left_toe_roll": 0.0,
        "idx07_right_hip_roll": 0.0,
        "idx08_right_hip_yaw": 0.0,
        "idx09_right_hip_pitch": -0.115,
        "idx10_right_tarsus": 0.267,
        "idx11_right_toe_pitch": -0.152,
        "idx12_right_toe_roll": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # Hips & legs
        "idx01_left_hip_roll": "effort",
        "idx02_left_hip_yaw": "effort",
        "idx03_left_hip_pitch": "effort",
        "idx04_left_tarsus": "effort",
        "idx05_left_toe_pitch": "effort",
        "idx06_left_toe_roll": "effort",
        "idx07_right_hip_roll": "effort",
        "idx08_right_hip_yaw": "effort",
        "idx09_right_hip_pitch": "effort",
        "idx10_right_tarsus": "effort",
        "idx11_right_toe_pitch": "effort",
        "idx12_right_toe_roll": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["left_toe_roll", "right_toe_roll"]
    knee_links: list[str] = ["left_tarsus", "right_tarsus"]
    torso_links: list[str] = ["base_link"]
    elbow_links: list[str] = [
        "left_arm_link05",
        "right_arm_link05",
    ]  # TODO(zhangyi): find elbow links and add them here
    terminate_contacts_links = [
        "base_link",
        "left_hip_roll",
        "left_hip_yaw",
        "left_hip_pitch",
        "left_tarsus",
        "left_toe_pitch",
        "left_toe_roll",
        "right_hip_roll",
        "right_hip_yaw",
        "right_hip_pitch",
        "right_tarsus",
        "right_toe_pitch",
        "right_toe_roll",
    ]
    penalized_contacts_links: list[str] = [
        "left_hip_roll",
        "left_hip_yaw",
        "left_hip_pitch",
        "left_tarsus",
        "left_toe_pitch",
        "left_toe_roll",
        "right_hip_roll",
        "right_hip_yaw",
        "right_hip_pitch",
        "right_tarsus",
        "right_toe_pitch",
        "right_toe_roll",
    ]

    # joint substrings, to find indices of joints.
    left_hip_yaw_roll_joints: list[str] = ["idx02_left_hip_yaw", "idx01_left_hip_roll"]
    right_hip_yaw_roll_joints: list[str] = ["idx08_right_hip_yaw", "idx07_right_hip_roll"]

    soft_joint_pos_limit_factor = 0.95
    # From default joint armature in XML
    armature: float = 0.01
