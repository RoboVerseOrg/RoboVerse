from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class FetchCfg(RobotCfg):
    name: str = "fetch"
    num_joints: int = 9
    urdf_path: str = "roboverse_data/robots/fetch/robots/fetch.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = False
    fix_base_link: bool = True
    actuators: dict[str, BaseActuatorCfg] = {
        "r_wheel_joint": BaseActuatorCfg(velocity_limit=17.4),
        "l_wheel_joint": BaseActuatorCfg(velocity_limit=17.4),
        "torso_lift_joint": BaseActuatorCfg(velocity_limit=0.1),
        "head_pan_joint": BaseActuatorCfg(velocity_limit=1.57),
        "head_tilt_joint": BaseActuatorCfg(velocity_limit=1.57),
        "shoulder_pan_joint": BaseActuatorCfg(velocity_limit=1.256),
        "shoulder_lift_joint": BaseActuatorCfg(velocity_limit=1.454),
        "upperarm_roll_joint": BaseActuatorCfg(velocity_limit=1.571),
        "elbow_flex_joint": BaseActuatorCfg(velocity_limit=1.521),
        "forearm_roll_joint": BaseActuatorCfg(velocity_limit=1.571),
        "wrist_flex_joint": BaseActuatorCfg(velocity_limit=2.268),
        "wrist_roll_joint": BaseActuatorCfg(velocity_limit=2.268),
        "r_gripper_finger_joint": BaseActuatorCfg(velocity_limit=0.05, is_ee=True),
        "l_gripper_finger_joint": BaseActuatorCfg(velocity_limit=0.05, is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "r_wheel_joint": (-1000, 1000),  # Continuous joint
        "l_wheel_joint": (-1000, 1000),  # Continuous joint
        "torso_lift_joint": (0.0, 0.38615),
        "head_pan_joint": (-1.57, 1.57),
        "head_tilt_joint": (-0.76, 1.45),
        "shoulder_pan_joint": (-1.6056, 1.6056),
        "shoulder_lift_joint": (-1.221, 1.518),
        "upperarm_roll_joint": (-10, 10),  # Continuous joint
        "elbow_flex_joint": (-2.251, 2.251),
        "forearm_roll_joint": (-10, 10),  # Continuous joint
        "wrist_flex_joint": (-2.16, 2.16),
        "wrist_roll_joint": (-10, 10),  # Continuous joint
        "r_gripper_finger_joint": (0.0, 0.05),
        "l_gripper_finger_joint": (0.0, 0.05),
    }
    ee_body_name: str = "gripper_link"

    # 2-finger gripper widths (both fingers, limits (0, 0.05)). These stay: they
    # are a valid gripper descriptor used by ``process_gripper_command``.
    gripper_open_q = [0.04, 0.04]
    gripper_close_q = [0.0, 0.0]

    # ``curobo_ref_cfg_name = "franka.yml"`` was wrong: Fetch is not a Franka, and
    # there is no Fetch curobo config to point at. Fetch also cannot use the
    # arm-first ``IKSolver`` as-is -- its actuator order is wheels/torso/head first
    # and the 7-DOF arm is in the middle, not the leading ``n_dof_ik`` block, so IK
    # would solve wheels/torso/head instead of the arm. Removed rather than left
    # mis-pointed; wiring Fetch IK needs a Fetch curobo config plus an arm-only
    # actuator ordering (maintainer input).
    # curobo_ref_cfg_name = "<fetch>.yml"
    # curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, 0.10312]
    # curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
