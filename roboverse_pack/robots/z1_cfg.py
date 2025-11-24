from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Z1Cfg(RobotCfg):
    """Configuration for the Unitree Z1 Robotic Arm.

    The Z1 is a 6-DOF lightweight robotic arm designed for research and development
    with high precision and agility.
    """

    name: str = "z1"
    num_joints: int = 6
    fix_base_link: bool = True

    # Asset paths
    urdf_path: str = "roboverse_data/robots/Unitree_Z1_Robotic_Arm/urdf/z1.urdf"
    usd_path: str = "roboverse_data/robots/Unitree_Z1_Robotic_Arm/usd/z1.usd"
    mjcf_path: str = "roboverse_data/robots/Unitree_Z1_Robotic_Arm/mjcf/z1.xml"

    # Physical properties
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    # isaacgym_read_mjcf: bool = False  # Use URDF for IsaacLab/IsaacGym, MJCF available as alternative

    # Actuator configuration - Based on Z1 joint specifications
    # Using gainprm and force settings from MJCF
    actuators: dict[str, BaseActuatorCfg] = {
        "joint1": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=3.1415, torque_limit=30),
        "joint2": BaseActuatorCfg(
            stiffness=1500, damping=150, velocity_limit=3.1415, torque_limit=60
        ),  # Higher torque for shoulder
        "joint3": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=3.1415, torque_limit=30),
        "joint4": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=3.1415, torque_limit=30),
        "joint5": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=3.1415, torque_limit=30),
        "joint6": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=3.1415, torque_limit=30),
    }

    # Joint limits - Based on Z1 actual joint limits from MJCF (radians)
    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.61799, 2.61799),  # ±150°
        "joint2": (0.0, 2.96706),  # 0° to 170°
        "joint3": (-2.87979, 0.0),  # -165° to 0°
        "joint4": (-1.51844, 1.51844),  # ±87°
        "joint5": (-1.3439, 1.3439),  # ±77°
        "joint6": (-2.79253, 2.79253),  # ±160°
    }

    # End-effector body name - Last link
    ee_body_name: str = "link06"

    # Default joint positions - Z1 standard home position
    # Based on keyframe in MJCF
    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": 0.785,  # ~45°
        "joint3": -0.261,  # ~-15°
        "joint4": -0.523,  # ~-30°
        "joint5": 0.0,
        "joint6": 0.0,
    }

    # Control type - All joints use position control
    control_type: dict[str, Literal["position", "effort"]] = {
        "joint1": "position",
        "joint2": "position",
        "joint3": "position",
        "joint4": "position",
        "joint5": "position",
        "joint6": "position",
    }

    # Z1 has no gripper by default, gripper can be added separately if needed
    # gripper_open_q = []
    # gripper_close_q = []
