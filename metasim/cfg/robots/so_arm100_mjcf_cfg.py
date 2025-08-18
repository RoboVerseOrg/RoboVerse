from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class SoArm100Cfg(BaseRobotCfg):
    """Cfg for the Universal Robots UR5e robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "so_arm100"
    num_joints: int = 6
    fix_base_link: bool = True

    # usd_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/usd/arx_l5.usd"
    # urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/Siasun_SO-ARM100/mjcf/so_arm100.xml"

    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "Rotation": BaseActuatorCfg(velocity_limit=2.175),
        "Pitch": BaseActuatorCfg(velocity_limit=2.175),
        "Elbow": BaseActuatorCfg(velocity_limit=2.175),
        "Wrist_Pitch": BaseActuatorCfg(velocity_limit=2.175),
        "Wrist_Roll": BaseActuatorCfg(velocity_limit=2.61),
        "Jaw": BaseActuatorCfg(velocity_limit=2.61),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "Rotation": (-2.618, 2.618),
        "Pitch": (-2.059, 2.0944),
        "Elbow": (-2.697, 0),
        "Wrist_Pitch": (-0.19198, 3.927),
        "Wrist_Roll": (-1.22, 1.22),
        "Jaw": (-1.69297, 3.14159),
    }

    ee_body_name: str = "tool0"

    default_joint_positions: dict[str, float] = {
        "Rotation": 0.0,
        "Pitch": -0.785398,
        "Elbow": 0.0,
        "Wrist_Pitch": -2.356194,
        "Wrist_Roll": 0.0,
        "Jaw": 1.570796,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "Rotation": "position",
        "Pitch": "position",
        "Elbow": "position",
        "Wrist_Pitch": "position",
        "Wrist_Roll": "position",
        "Jaw": "position",
    }
