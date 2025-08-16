from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class Lite6Cfg(BaseRobotCfg):
    """Cfg for the Universal Robots UR5e robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "lite6"
    num_joints: int = 6
    fix_base_link: bool = True

    # usd_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/usd/arx_l5.usd"
    # urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/UFactory_Lite6/mjcf/lite6.xml"

    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "joint1": BaseActuatorCfg(velocity_limit=2.175),
        "joint2": BaseActuatorCfg(velocity_limit=2.175),
        "joint3": BaseActuatorCfg(velocity_limit=2.175),
        "joint4": BaseActuatorCfg(velocity_limit=2.175),
        "joint5": BaseActuatorCfg(velocity_limit=2.61),
        "joint6": BaseActuatorCfg(velocity_limit=2.61),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-6.28319, 6.28319),
        "joint2": (-2.61799, 2.61799),
        "joint3": (-0.061087, 5.23599),
        "joint4": (-6.28319, 6.28319),
        "joint5": (-2.1642, 2.1642),
        "joint6": (-6.28319, 6.28319),
    }

    ee_body_name: str = "tool0"

    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": -0.785398,
        "joint3": 0.0,
        "joint4": -2.356194,
        "joint5": 0.0,
        "joint6": 1.570796,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "joint1": "position",
        "joint2": "position",
        "joint3": "position",
        "joint4": "position",
        "joint5": "position",
        "joint6": "position",
    }
