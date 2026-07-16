from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class IiwaCfg(RobotCfg):
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
    # Joint limits are required: ``IKSolver`` reads ``list(joint_limits.keys())``,
    # so a None here raised ``AttributeError`` at construction. Arm limits are the
    # physical KUKA LBR iiwa 7 ranges (A1/3/5 = +/-170 deg, A2/4/6 = +/-120 deg,
    # A7 = +/-175 deg), matching the robosuite iiwa asset
    # (robosuite/models/assets/robots/iiwa/robot.xml). The panda finger limits
    # (0, 0.04) are the Franka hand ranges from the franka_panda.urdf shipped in
    # roboverse_data (this asset mounts a Franka hand on the iiwa7).
    joint_limits: dict[str, tuple[float, float]] = {
        "iiwa7_joint_1": (-2.96706, 2.96706),
        "iiwa7_joint_2": (-2.0944, 2.0944),
        "iiwa7_joint_3": (-2.96706, 2.96706),
        "iiwa7_joint_4": (-2.0944, 2.0944),
        "iiwa7_joint_5": (-2.96706, 2.96706),
        "iiwa7_joint_6": (-2.0944, 2.0944),
        "iiwa7_joint_7": (-3.05433, 3.05433),
        "panda_finger_joint1": (0.0, 0.04),
        "panda_finger_joint2": (0.0, 0.04),
    }
    ee_body_name: str = "panda_hand"
    gripper_open_q = [0.04, 0.04]
    gripper_close_q = [0.0, 0.0]

    curobo_ref_cfg_name: str = "iiwa.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.00074, 0.10312]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
