from __future__ import annotations

import math

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class KinovaGen3Cfg(RobotCfg):
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

    # This is the bare 7-DOF Kinova Gen3 arm: no gripper joints are actuated. The
    # IK wiring here was copied from Franka and was wrong on both counts. The
    # 2-element ``gripper_open_q`` (Franka finger widths) made
    # ``n_dof_ik = len(actuators) - len(gripper_open_q) = 7 - 2 = 5``, so IK solved
    # only 5 of the 7 arm joints and wrote gripper widths onto joint_6/joint_7.
    # ``franka.yml`` is Franka's kinematics, not this arm. Both are removed so
    # ``IKSolver`` is not mis-constructed. To run IK, add a real Kinova Gen3
    # curobo config (the arm+gripper sibling uses ``kinova_gen3.yml``, but that
    # includes the 2F-85 and may not fit this bare arm) or a ``urdf_path`` for the
    # pyroki backend; add a gripper cfg if grasping is needed.
    # gripper_open_q = [...]
    # gripper_close_q = [...]
    # curobo_ref_cfg_name = "<kinova_gen3_arm>.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
