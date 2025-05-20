from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class SawyerCfg(BaseRobotCfg):
    name: str = "sawyer"
    num_joints: int = 10
    usd_path: str = "roboverse_data/robots/sawyer/usd/sawyer_v2.usd"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "right_j0": BaseActuatorCfg(),
        "right_j1": BaseActuatorCfg(),
        "right_j2": BaseActuatorCfg(),
        "right_j3": BaseActuatorCfg(),
        "right_j4": BaseActuatorCfg(),
        "right_j5": BaseActuatorCfg(),
        "right_j6": BaseActuatorCfg(),
        "head_pan": BaseActuatorCfg(),
        "right_gripper_l_finger_joint": BaseActuatorCfg(is_ee=True),
        "right_gripper_r_finger_joint": BaseActuatorCfg(is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "right_j0": (-3.05, 3.05),
        "right_j1": (-3.8094999, 2.2736),
        "right_j2": (-3.0425998, 3.0425998),
        "right_j3": (-3.0438999, 3.0438999),
        "right_j4": (-2.9760998, 2.9760998),
        "right_j5": (-2.9760998, 2.9760998),
        "right_j6": (-4.7123996, 4.7123996),
        "head_pan": (-5.095, 0.9064),
        "right_gripper_l_finger_joint": (0.0, 0.020833),
        "right_gripper_r_finger_joint": (-0.020833, 0),
    }
    ee_body_name: str = "sawyer_right_hand"
    gripper_open_q = [0.020833, -0.020833]
    gripper_close_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "sawyer.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.105]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
