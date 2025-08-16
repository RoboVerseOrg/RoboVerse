from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class ArxL5Cfg(BaseRobotCfg):
    """Cfg for the Universal Robots UR5e robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "arx_l5"
    num_joints: int = 8
    fix_base_link: bool = True

    # usd_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/usd/arx_l5.usd"
    # urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/mjcf/arx_l5.xml"

    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "joint6": BaseActuatorCfg(stiffness=250, damping=50, velocity_limit=2.61),
        "gripper": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=0.2, is_ee=True),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-3.14, 3.14),
        "joint2": (0, 3.14),
        "joint3": (0, 3.14),
        "joint4": (-1.7, 1.7),
        "joint5": (-1.7, 1.7),
        "joint6": (-3.14, 3.14),
        "gripper": (0, 0.044),
        "joint8": (-0.044, 0),
    }

    ee_body_name: str = "tool0"

    default_joint_positions: dict[str, float] = {
        "joint1": 0.0,
        "joint2": -0.785398,
        "joint3": 0.0,
        "joint4": -2.356194,
        "joint5": 0.0,
        "joint6": 1.570796,
        "gripper": 0.044,
        "joint8": -0.044,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "joint1": "position",
        "joint2": "position",
        "joint3": "position",
        "joint4": "position",
        "joint5": "position",
        "joint6": "position",
        "gripper": "position",
        "joint8": "position",
    }

    gripper_open_q = [-0.044, 0.044]
    gripper_close_q = [0.0, 0.0]

    # curobo_ref_cfg_name: str = "ur5e.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.0]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
