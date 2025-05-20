from __future__ import annotations

import math

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class KinovaGen3Cfg(BaseRobotCfg):
    """Cfg for the Kinova Gen3 robot."""

    name: str = "kinova_gen3"
    num_joints: int = 9
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/kinova_gen3/usd/kinova_gen3_v1.usd"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "joint_1": BaseActuatorCfg(),
        "joint_2": BaseActuatorCfg(),
        "joint_3": BaseActuatorCfg(),
        "joint_4": BaseActuatorCfg(),
        "joint_5": BaseActuatorCfg(),
        "joint_6": BaseActuatorCfg(),
        "joint_7": BaseActuatorCfg(),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "joint_1": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_2": (-2.4100, 2.4100),
        "joint_3": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_4": (-2.6600, 2.6600),
        "joint_5": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_6": (-2.2300, 2.2300),
        "joint_7": (-math.pi, math.pi),  # actually is -inf to +inf
    }
    ee_body_name: str = "end_effector_link"

    gripper_open_q = [0.04, 0.04]  # TODO
    gripper_close_q = [0.0, 0.0]  # TODO

    curobo_ref_cfg_name: str = "franka.yml"  # TODO
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]  # TODO
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]  # TODO
