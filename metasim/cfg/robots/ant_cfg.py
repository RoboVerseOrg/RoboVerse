from __future__ import annotations

import math

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AntCfg(BaseRobotCfg):
    name: str = "ant"
    num_joints: int = 8
    fix_base_link: bool = False
    usd_path: str = "roboverse_data/robots/ant/usd/ant.usd"
    mjcf_path: str = "roboverse_data/robots/ant/mjcf/ant.xml"
    urdf_path: str = "roboverse_data/robots/ant/urdf/ant.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    # Define actuators for each joint - control range from -1.0 to 1.0 with gear 15
    actuators: dict[str, BaseActuatorCfg] = {
        "hip_1": BaseActuatorCfg(velocity_limit=30.0),
        "ankle_1": BaseActuatorCfg(velocity_limit=30.0),
        "hip_2": BaseActuatorCfg(velocity_limit=30.0),
        "ankle_2": BaseActuatorCfg(velocity_limit=30.0),
        "hip_3": BaseActuatorCfg(velocity_limit=30.0),
        "ankle_3": BaseActuatorCfg(velocity_limit=30.0),
        "hip_4": BaseActuatorCfg(velocity_limit=30.0),
        "ankle_4": BaseActuatorCfg(velocity_limit=30.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "hip_1": (-40 * math.pi / 180, 40 * math.pi / 180),  # -40 to 40 degrees
        "ankle_1": (30 * math.pi / 180, 100 * math.pi / 180),  # 30 to 100 degrees
        "hip_2": (-40 * math.pi / 180, 40 * math.pi / 180),  # -40 to 40 degrees
        "ankle_2": (-100 * math.pi / 180, -30 * math.pi / 180),  # -100 to -30 degrees
        "hip_3": (-40 * math.pi / 180, 40 * math.pi / 180),  # -40 to 40 degrees
        "ankle_3": (-100 * math.pi / 180, -30 * math.pi / 180),  # -100 to -30 degrees
        "hip_4": (-40 * math.pi / 180, 40 * math.pi / 180),  # -40 to 40 degrees
        "ankle_4": (30 * math.pi / 180, 100 * math.pi / 180),  # 30 to 100 degrees
    }

    default_joint_positions: dict[str, float] = {
        "hip_1": 0.0,
        "ankle_1": 1.0,
        "hip_2": 0.0,
        "ankle_2": -1.0,
        "hip_3": 0.0,
        "ankle_3": -1.0,
        "hip_4": 0.0,
        "ankle_4": 1.0,
    }

    control_frequency_inv: int = 1  # Control frequency divider
    motor_strength: float = 15.0  # Motor gear value from XML
    dof_damping: float = 0.1  # From default joint damping in XML
    dof_friction: float = 0.0  # Not explicitly defined in XML
    dof_armature: float = 0.01  # From default joint armature in XML

    observe_base_position: bool = True
    observe_base_velocity: bool = True
    observe_joint_positions: bool = True
    observe_joint_velocities: bool = True
