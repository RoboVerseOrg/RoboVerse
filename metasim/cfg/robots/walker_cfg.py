from __future__ import annotations

import math

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class WalkerCfg(BaseRobotCfg):
    name: str = "walker"
    num_joints: int = 6
    fix_base_link: bool = False
    usd_path: str = "roboverse_data/robots/walker/usd/walker.usd"
    mjcf_path: str = "roboverse_data/robots/walker/mjcf/walker.xml"
    urdf_path: str = "roboverse_data/robots/walker/urdf/walker.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    # Define actuators for each joint - control range from -1.0 to 1.0 with gear values from XML
    actuators: dict[str, BaseActuatorCfg] = {
        "right_hip": BaseActuatorCfg(velocity_limit=30.0),
        "right_knee": BaseActuatorCfg(velocity_limit=30.0),
        "right_ankle": BaseActuatorCfg(velocity_limit=30.0),
        "left_hip": BaseActuatorCfg(velocity_limit=30.0),
        "left_knee": BaseActuatorCfg(velocity_limit=30.0),
        "left_ankle": BaseActuatorCfg(velocity_limit=30.0),
    }

    # Joint limits from the XML (converted from degrees to radians)
    joint_limits: dict[str, tuple[float, float]] = {
        "right_hip": (-20 * math.pi / 180, 100 * math.pi / 180),  # -20 to 100 degrees
        "right_knee": (-150 * math.pi / 180, 0 * math.pi / 180),  # -150 to 0 degrees
        "right_ankle": (-45 * math.pi / 180, 45 * math.pi / 180),  # -45 to 45 degrees
        "left_hip": (-20 * math.pi / 180, 100 * math.pi / 180),  # -20 to 100 degrees
        "left_knee": (-150 * math.pi / 180, 0 * math.pi / 180),  # -150 to 0 degrees
        "left_ankle": (-45 * math.pi / 180, 45 * math.pi / 180),  # -45 to 45 degrees
    }

    # Default starting pose - using zeros (no explicit init_qpos in the XML)
    default_joint_positions: dict[str, float] = {
        "right_hip": 0.0,
        "right_knee": 0.0,
        "right_ankle": 0.0,
        "left_hip": 0.0,
        "left_knee": 0.0,
        "left_ankle": 0.0,
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 1.3)

    # Control constants from the XML
    control_frequency_inv: int = 1  # Control frequency divider
    motor_strength: dict[str, float] = {
        "right_hip": 100.0,  # From gear="100" in XML
        "right_knee": 50.0,  # From gear="50" in XML
        "right_ankle": 20.0,  # From gear="20" in XML
        "left_hip": 100.0,  # From gear="100" in XML
        "left_knee": 50.0,  # From gear="50" in XML
        "left_ankle": 20.0,  # From gear="20" in XML
    }

    # From default joint damping in XML
    dof_damping: float = 0.1
    # From default joint friction in XML
    dof_friction: float = 0.0
    # From default joint armature in XML
    dof_armature: float = 0.01

    # Observation space configuration
    observe_base_position: bool = True
    observe_base_velocity: bool = True
    observe_joint_positions: bool = True
    observe_joint_velocities: bool = True
