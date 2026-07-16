from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Ur5E2F85Cfg(RobotCfg):
    name: str = "ur5e_2f85"
    num_joints: int = 12
    usd_path: str = "data_isaaclab/robots/UniversalRobots/ur5e/ur5e_2f85_fix.usd"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "shoulder_pan_joint": BaseActuatorCfg(velocity_limit=2.175),
        "shoulder_lift_joint": BaseActuatorCfg(velocity_limit=2.175),
        "elbow_joint": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_1_joint": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_2_joint": BaseActuatorCfg(velocity_limit=2.61),
        "wrist_3_joint": BaseActuatorCfg(velocity_limit=2.61),
        # "finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "left_inner_finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "left_inner_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_inner_finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_inner_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_outer_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "shoulder_pan_joint": (-6.28319, 6.28319),
        "shoulder_lift_joint": (-6.28319, 6.28319),
        "elbow_joint": (-3.14159, 3.14159),
        "wrist_1_joint": (-6.28319, 6.28319),
        "wrist_2_joint": (-6.28319, 6.28319),
        "wrist_3_joint": (-6.28319, 6.28319),
        # "finger_joint": (0.0, 0.785398),
        # "left_inner_finger_joint": (0.0, 0.785398),
        # "left_inner_knuckle_joint": (0.0, 0.785398),
        # "right_inner_finger_joint": (0.0, 0.785398),
        # "right_inner_knuckle_joint": (0.0, 0.785398),
        # "right_outer_knuckle_joint": (0.0, 0.785398),
    }
    ee_body_name: str = "wrist_3_link"
    # The Robotiq 2F-85 gripper joints are commented out of ``actuators`` and
    # ``joint_limits`` above, so this cfg exposes only the 6-DOF UR5e arm. The IK
    # wiring that used to live here was left over from that removal and was self-
    # inconsistent: ``gripper_open_q`` listed 6 gripper widths while 0 gripper
    # joints are actuated, so ``n_dof_ik = len(actuators) - len(gripper_open_q)``
    # was ``6 - 6 = 0`` (IK would solve zero arm joints). ``ur5e_robotiq_2f_140.yml``
    # is also the 140 mm gripper, not this 2F-85. It is removed so ``IKSolver`` is
    # not mis-constructed. To run IK + grasping, re-actuate the gripper joints and
    # point ``curobo_ref_cfg_name`` at a real UR5e + 2F-85 config.
    # gripper_open_q = [...]
    # gripper_close_q = [...]
    # curobo_ref_cfg_name = "<ur5e_2f85>.yml"
