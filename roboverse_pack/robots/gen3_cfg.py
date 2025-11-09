from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Gen3Cfg(RobotCfg):
    """Configuration for the Kinova Gen3 Robotic Arm.

    The Gen3 is a 7-DOF lightweight robotic arm designed for collaborative robotics
    applications with advanced sensing and control capabilities.
    """

    name: str = "gen3"
    num_joints: int = 7
    fix_base_link: bool = True

    # Asset paths
    urdf_path: str = "roboverse_data/robots/Kinova_Gen3/urdf/gen3.urdf"
    usd_path: str = "roboverse_data/robots/Kinova_Gen3/usd/gen3.usd"
    mjcf_path: str = "roboverse_data/robots/Kinova_Gen3/mjcf/gen3.xml"

    # Physical properties
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    # isaacgym_read_mjcf: bool = False  # Use URDF for IsaacLab/IsaacGym, MJCF available as alternative

    # Actuator configuration - Based on Gen3 joint specifications
    # Large actuators for main arm joints, small actuators for wrist joints
    actuators: dict[str, BaseActuatorCfg] = {
        "joint_1": BaseActuatorCfg(stiffness=2000, damping=100, velocity_limit=1.3963, torque_limit=105),
        "joint_2": BaseActuatorCfg(stiffness=2000, damping=100, velocity_limit=1.3963, torque_limit=105),
        "joint_3": BaseActuatorCfg(stiffness=2000, damping=100, velocity_limit=1.3963, torque_limit=105),
        "joint_4": BaseActuatorCfg(stiffness=2000, damping=100, velocity_limit=1.3963, torque_limit=105),
        "joint_5": BaseActuatorCfg(stiffness=500, damping=50, velocity_limit=1.2218, torque_limit=52),  # Small actuator
        "joint_6": BaseActuatorCfg(stiffness=500, damping=50, velocity_limit=1.2218, torque_limit=52),  # Small actuator
        "joint_7": BaseActuatorCfg(stiffness=500, damping=50, velocity_limit=1.2218, torque_limit=52),  # Small actuator
    }

    # Joint limits - Based on Gen3 actual joint limits from MJCF/URDF (radians)
    # Some joints are continuous (unlimited rotation)
    joint_limits: dict[str, tuple[float, float]] = {
        "joint_1": (-6.28319, 6.28319),  # Continuous joint (~±360°)
        "joint_2": (-2.41, 2.41),  # ±138°
        "joint_3": (-6.28319, 6.28319),  # Continuous joint (~±360°)
        "joint_4": (-2.66, 2.66),  # ±152°
        "joint_5": (-6.28319, 6.28319),  # Continuous joint (~±360°)
        "joint_6": (-2.23, 2.23),  # ±128°
        "joint_7": (-6.28319, 6.28319),  # Continuous joint (~±360°)
    }

    # End-effector body name - Bracelet link with pinch site
    ee_body_name: str = "bracelet_link"

    # Default joint positions - Gen3 standard home position
    # Based on keyframe in MJCF
    default_joint_positions: dict[str, float] = {
        "joint_1": 0.0,
        "joint_2": 0.262,  # ~15°
        "joint_3": 3.142,  # ~180°
        "joint_4": -2.269,  # ~-130°
        "joint_5": 0.0,
        "joint_6": 0.960,  # ~55°
        "joint_7": 1.571,  # ~90°
    }

    # Control type - All joints use position control
    control_type: dict[str, Literal["position", "effort"]] = {
        "joint_1": "position",
        "joint_2": "position",
        "joint_3": "position",
        "joint_4": "position",
        "joint_5": "position",
        "joint_6": "position",
        "joint_7": "position",
    }

    # Gen3 has no gripper by default, gripper can be added separately if needed
    # gripper_open_q = []
    # gripper_close_q = []
