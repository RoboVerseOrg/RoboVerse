from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class IiwaCfg(BaseRobotCfg):
    name: str = "iiwa"
    num_joints: int = 9
    usd_path: str = "data_isaaclab/robots/iiwa/iiwa.usd"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "iiwa7_joint_1": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_2": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_3": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_4": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_5": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_6": BaseActuatorCfg(velocity_limit=None),
        "iiwa7_joint_7": BaseActuatorCfg(velocity_limit=None),
        "panda_finger_joint1": BaseActuatorCfg(velocity_limit=None, is_ee=True),
        "panda_finger_joint2": BaseActuatorCfg(velocity_limit=None, is_ee=True),
    }
    ee_body_name: str = "panda_hand"
    gripper_open_q = [0.04, 0.04]
    gripper_close_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "iiwa.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.00074, 0.10312]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
