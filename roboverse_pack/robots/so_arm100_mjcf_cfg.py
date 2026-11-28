from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class SoArm100Cfg(RobotCfg):
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

    # The ranges the MJCF declares (``so_arm100.xml`` ``<default class=...>`` joints); the previous values were
    # another arm's and put the Wrist_Pitch default 0.7 rad outside the reachable range (robot check, 2026-11-28).
    joint_limits: dict[str, tuple[float, float]] = {
        "Rotation": (-1.92, 1.92),
        "Pitch": (-3.32, 0.174),
        "Elbow": (-0.174, 3.14),
        "Wrist_Pitch": (-1.66, 1.66),
        "Wrist_Roll": (-2.79, 2.79),
        "Jaw": (-0.174, 1.75),
    }

    ee_body_name: str = (
        "Fixed_Jaw"  # the MJCF has no "tool0"; this is its last link before the tool (robot check, 2026-11-28)
    )

    default_joint_positions: dict[str, float] = {
        "Rotation": 0.0,
        "Pitch": -0.785398,
        "Elbow": 0.0,
        "Wrist_Pitch": -1.570796,  # the nearest right angle inside the MJCF range (-2.356 was unreachable)
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
