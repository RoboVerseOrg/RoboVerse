from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class YamCfg(BaseRobotCfg):
    """Cfg for the Universal Robots UR5e robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "yam"
    num_joints: int = 7
    fix_base_link: bool = True

    # usd_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/usd/arx_l5.usd"
    # urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/Yet_Another_Manipulator_YAM/mjcf/yam.xml"

    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "joint1": BaseActuatorCfg(velocity_limit=2.175),
        "joint2": BaseActuatorCfg(velocity_limit=2.175),
        "joint3": BaseActuatorCfg(velocity_limit=2.175),
        "joint4": BaseActuatorCfg(velocity_limit=2.175),
        "joint5": BaseActuatorCfg(velocity_limit=2.61),
        "joint6": BaseActuatorCfg(velocity_limit=2.61),
        "left_finger": BaseActuatorCfg(velocity_limit=0.2, is_ee=True),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.61799, 3.05433),
        "joint2": (0, 3.66519),
        "joint3": (0, 3.66519),
        "joint4": (-1.5708, 1.5708),
        "joint5": (-1.5708, 1.5708),
        "joint6": (-2.0944, 2.0944),
        "left_finger": (-0.00205, 0.037524),
    }

    ee_body_name: str = "tool0"

    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": -0.785398,
        "joint3": 0.0,
        "joint4": -2.356194,
        "joint5": 0.0,
        "joint6": 1.570796,
        "left_finger": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "joint1": "position",
        "joint2": "position",
        "joint3": "position",
        "joint4": "position",
        "joint5": "position",
        "joint6": "position",
        "left_finger": "position",
    }

    gripper_open_q = [0.037524, 0.037524]
    gripper_close_q = [-0.00205, -0.00205]

    # curobo_ref_cfg_name: str = "ur5e.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.0]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
