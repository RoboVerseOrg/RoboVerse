from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Ur5ECfg(RobotCfg):
    """Configuration for the Universal Robots UR5e robot.

    The UR5e is a 6-DOF industrial robotic arm with medium payload capacity,
    widely used for collaborative applications and automation tasks.
    """

    name: str = "ur5e"
    num_joints: int = 6
    fix_base_link: bool = True

    # Asset paths
    usd_path: str = "roboverse_data/robots/Universal_Robots_UR5e/usd/ur5e.usd"
    urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/mjcf/ur5e.xml"

    # Physical properties
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    # isaacgym_read_mjcf: bool = False  # Use URDF for IsaacLab/IsaacGym

    # Actuator configuration - Based on UR5e joint specifications
    actuators: dict[str, BaseActuatorCfg] = {
        "shoulder_pan_joint": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=3.15),
        "shoulder_lift_joint": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=3.15),
        "elbow_joint": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=3.15),
        "wrist_1_joint": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=6.28),
        "wrist_2_joint": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=6.28),
        "wrist_3_joint": BaseActuatorCfg(stiffness=250, damping=50, velocity_limit=6.28),
    }

    # Joint limits - Based on UR5e actual joint limits (radians)
    joint_limits: dict[str, tuple[float, float]] = {
        "shoulder_pan_joint": (-6.28318, 6.28318),  # ±360°
        "shoulder_lift_joint": (-6.28318, 6.28318),  # ±360°
        "elbow_joint": (-3.14159, 3.14159),  # ±180°
        "wrist_1_joint": (-6.28318, 6.28318),  # ±360°
        "wrist_2_joint": (-6.28318, 6.28318),  # ±360°
        "wrist_3_joint": (-6.28318, 6.28318),  # ±360°
    }

    # End-effector body name - UR5e tool flange
    ee_body_name: str = "tool0"

    # Default joint positions - UR5e standard home position
    default_joint_positions: dict[str, float] = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.5708,  # -90°
        "elbow_joint": 0.0,
        "wrist_1_joint": -1.5708,  # -90°
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    }

    # Control type - All joints use position control
    control_type: dict[str, Literal["position", "effort"]] = {
        "shoulder_pan_joint": "position",
        "shoulder_lift_joint": "position",
        "elbow_joint": "position",
        "wrist_1_joint": "position",
        "wrist_2_joint": "position",
        "wrist_3_joint": "position",
    }

    # UR5e has no gripper by default, gripper can be added separately if needed
    # gripper_open_q = []
    # gripper_close_q = []
