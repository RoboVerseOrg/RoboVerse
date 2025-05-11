from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class FrankaCfg(BaseRobotCfg):
    """Cfg for the Franka Emika Panda robot.

    Args:
        BaseRobotCfg (_type_): _description_
    """

    name: str = "franka"
    num_joints: int = 9
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/franka/usd/franka_v2.usd"
    mjcf_path: str = "roboverse_data/robots/franka/mjcf/panda.xml"
    # urdf_path: str = "roboverse_data/robots/franka/urdf/panda.urdf"  # work for pybullet and sapien
    urdf_path: str = "roboverse_data/robots/franka/urdf/franka_panda.urdf"  # work for isaacgym
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=250, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
        "panda_finger_joint1": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=0.2, is_ee=True),
        "panda_finger_joint2": BaseActuatorCfg(stiffness=1000, damping=100, velocity_limit=0.2, is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
        "panda_finger_joint1": (0.0, 0.04),  # 0.0 close, 0.04 open
        "panda_finger_joint2": (0.0, 0.04),  # 0.0 close, 0.04 open
    }
    ee_body_name: str = "panda_hand"

    default_joint_positions: dict[str, float] = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 1.570796,
        "panda_joint7": 0.785398,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    control_type: dict[str, Literal["position", "effort"]] = {
        "panda_joint1": "position",
        "panda_joint2": "position",
        "panda_joint3": "position",
        "panda_joint4": "position",
        "panda_joint5": "position",
        "panda_joint6": "position",
        "panda_joint7": "position",
        "panda_finger_joint1": "position",
        "panda_finger_joint2": "position",
    }

    # TODO: Make it more elegant
    gripper_release_q = [0.04, 0.04]
    gripper_actuate_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "franka.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
