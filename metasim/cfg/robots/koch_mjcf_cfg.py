from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class KochCfg(BaseRobotCfg):
    """Cfg for the Universal Robots UR5e robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "koch_1"
    num_joints: int = 6
    fix_base_link: bool = True

    # usd_path: str = "roboverse_data/robots/ARX_Robotics_L5_Arm/usd/arx_l5.usd"
    # urdf_path: str = "roboverse_data/robots/Universal_Robots_UR5e/urdf/ur_description/urdf/ur5e.urdf"
    mjcf_path: str = "roboverse_data/robots/Koch_v1.1_Low-Cost_Robot/mjcf/low_cost_robot_arm.xml"

    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "base_rotation": BaseActuatorCfg(velocity_limit=2.175),
        "pitch": BaseActuatorCfg(velocity_limit=2.175),
        "elbow": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_pitch": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_roll": BaseActuatorCfg(velocity_limit=2.61),
        "gripper": BaseActuatorCfg(velocity_limit=0.2, is_ee=True),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "base_rotation": (-2.618, 2.618),
        "pitch": (-2.059, 2.0944),
        "elbow": (-2.697, 0),
        "wrist_pitch": (-0.19198, 3.927),
        "wrist_roll": (-1.22, 1.22),
        "gripper": (0, 0.035),
    }

    ee_body_name: str = "tool0"

    default_joint_positions: dict[str, float] = {
        "base_rotation": 0.0,
        "pitch": -0.785398,
        "elbow": 0.0,
        "wrist_pitch": -2.356194,
        "wrist_roll": 0.0,
        "gripper": 0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "base_rotation": "position",
        "pitch": "position",
        "elbow": "position",
        "wrist_pitch": "position",
        "wrist_roll": "position",
        "gripper": "position",
    }

    gripper_open_q = [0.035, 0.035]
    gripper_close_q = [0.0, 0.0]

    # curobo_ref_cfg_name: str = "ur5e.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.0]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
