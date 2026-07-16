from __future__ import annotations

import math

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class KinovaGen3Robotiq2f85Cfg(RobotCfg):
    """Cfg for the Kinova Gen3 arm with Robotiq 2F-85 gripper."""

    name: str = "kinova_gen3_robotiq_2f85"
    num_joints: int = 9
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/kinova_gen3_robotiq_2f85/usd/kinova_gen3_robotiq_2f85_v1.usd"
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
        "finger_joint": BaseActuatorCfg(),
        "left_outer_finger_joint": BaseActuatorCfg(fully_actuated=False),
        "right_outer_finger_joint": BaseActuatorCfg(fully_actuated=False),
    }
    # ``joint_limits`` must list exactly the actuated joints, in ``actuators``
    # order: ``IKSolver`` uses ``len(joint_limits)`` as the robot DOF count and
    # ``compose_joint_action`` writes the gripper widths to the trailing
    # ``ee_n_dof`` columns and then zips them against ``actuators.keys()``. The old
    # 15-entry dict (7 arm joints plus 8 assorted passive 2F-85 linkage joints, in
    # a different order than ``actuators``) put the trailing columns on passive
    # joints, so the gripper command was silently dropped. The five passive
    # linkage joints (right_outer_knuckle, inner-finger, inner-knuckle) are driven
    # by the URDF/USD mimic couplings and are not listed here.
    joint_limits: dict[str, tuple[float, float]] = {
        "joint_1": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_2": (-2.4100, 2.4100),
        "joint_3": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_4": (-2.6600, 2.6600),
        "joint_5": (-math.pi, math.pi),  # actually is -inf to +inf
        "joint_6": (-2.2300, 2.2300),
        "joint_7": (-math.pi, math.pi),  # actually is -inf to +inf
        "finger_joint": (0.0000, 0.7854),
        "left_outer_finger_joint": (0.0000, 3.1416),
        "right_outer_finger_joint": (0.0000, 3.1416),
    }
    ee_body_name: str = "end_effector_link"

    # Three trailing gripper joints -> length-3 gripper qs, so
    # ``n_dof_ik = 10 - 3 = 7`` (the arm DOF). The gripper is driven by
    # ``finger_joint`` (open 0.0, close 0.7854); ``left/right_outer_finger_joint``
    # are underactuated (``fully_actuated=False``) mimic joints whose targets the
    # backends ignore, so their open/close entries are neutral 0.0 placeholders.
    gripper_open_q = [0.0, 0.0, 0.0]
    gripper_close_q = [0.7854, 0.0, 0.0]
    curobo_ref_cfg_name: str = "kinova_gen3.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.125]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
