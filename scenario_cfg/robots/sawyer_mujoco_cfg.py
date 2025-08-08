from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class SawyerMujocoCfg(BaseRobotCfg):
    name: str = "sawyer"
    num_joints: int = 9
    usd_path: str = "roboverse_data/robots/sawyer/usd/sawyer_mujoco_v1.usd"
    mjcf_path: str = "roboverse_data/robots/sawyer/mjcf/sawyer.xml"
    fix_base_link: bool = True
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
        "r_close": BaseActuatorCfg(is_ee=True),
        "l_close": BaseActuatorCfg(is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "right_j0": (-3.0503, 3.0503),
        "right_j1": (-3.8, -0.5),
        "right_j2": (-3.0426, 3.0426),
        "right_j3": (-3.0439, 3.0439),
        "right_j4": (-2.9761, 2.9761),
        "right_j5": (-2.9761, 2.9761),
        "right_j6": (-4.7124, 4.7124),
        "r_close": (0.0, 0.04),
        "l_close": (-0.03, 0),
    }
    # ee_body_name: str = "sawyer_right_hand"
    # gripper_open_q = [0.020833, -0.020833]
    # gripper_close_q = [0.0, 0.0]

    # curobo_ref_cfg_name: str = "sawyer.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.105]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
