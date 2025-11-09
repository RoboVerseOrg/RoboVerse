from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Iiwa14Cfg(RobotCfg):
    """Configuration for the KUKA LBR IIWA 14 R820 robot.

    The IIWA 14 is a 7-DOF lightweight robotic arm designed for
    human-robot collaboration with a 14kg payload capacity.
    """

    name: str = "iiwa14"
    num_joints: int = 7
    fix_base_link: bool = True

    # Asset paths
    urdf_path: str = "roboverse_data/robots/KUKA_LBR_IIWA14/urdf/iiwa.urdf"
    usd_path: str = "roboverse_data/robots/KUKA_LBR_IIWA14/usd/iiwa.usd"
    mjcf_path: str = "roboverse_data/robots/KUKA_LBR_IIWA14/mjcf/iiwa14.xml"

    # Physical properties
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    # isaacgym_read_mjcf: bool = False  # Use URDF for IsaacLab/IsaacGym, MJCF available as alternative

    # Actuator configuration - Based on IIWA14 joint specifications
    # Using the gainprm and bias settings from MJCF default classes
    actuators: dict[str, BaseActuatorCfg] = {
        "joint1": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=1.48),  # ~85 deg/s
        "joint2": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=1.48),  # ~85 deg/s
        "joint3": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=1.75),  # ~100 deg/s
        "joint4": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=1.31),  # ~75 deg/s
        "joint5": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=2.27),  # ~130 deg/s
        "joint6": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=2.36),  # ~135 deg/s
        "joint7": BaseActuatorCfg(stiffness=2000, damping=200, velocity_limit=2.36),  # ~135 deg/s
    }

    # Joint limits - Based on IIWA14 actual joint limits from MJCF (radians)
    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.96706, 2.96706),  # ±170°
        "joint2": (-2.0944, 2.0944),  # ±120°
        "joint3": (-3.05433, 3.05433),  # ±175°
        "joint4": (-2.0944, 2.0944),  # ±120°
        "joint5": (-2.96706, 2.96706),  # ±170°
        "joint6": (-2.0944, 2.0944),  # ±120°
        "joint7": (-3.05433, 3.05433),  # ±175°
    }

    # End-effector body name - Attachment site on link7
    ee_body_name: str = "link7"

    # Default joint positions - IIWA14 standard home position
    # Based on the commented keyframe in MJCF
    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": 0.785398,  # 45°
        "joint3": 0.0,
        "joint4": -1.5708,  # -90°
        "joint5": 0.0,
        "joint6": 0.0,
        "joint7": 0.0,
    }

    # Control type - All joints use position control
    control_type: dict[str, Literal["position", "effort"]] = {
        "joint1": "position",
        "joint2": "position",
        "joint3": "position",
        "joint4": "position",
        "joint5": "position",
        "joint6": "position",
        "joint7": "position",
    }

    # IIWA14 has no gripper by default, gripper can be added separately if needed
    # gripper_open_q = []
    # gripper_close_q = []
