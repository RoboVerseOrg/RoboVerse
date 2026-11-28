from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class GoogleRobotStaticCfg(RobotCfg):
    name: str = "google_robot_static"
    num_joints: int = 11
    urdf_path: str = "roboverse_data/robots/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "joint_torso": BaseActuatorCfg(velocity_limit=0.5),
        "joint_shoulder": BaseActuatorCfg(velocity_limit=0.5),
        "joint_bicep": BaseActuatorCfg(velocity_limit=0.5),
        "joint_elbow": BaseActuatorCfg(velocity_limit=0.5),
        "joint_forearm": BaseActuatorCfg(velocity_limit=0.5),
        "joint_wrist": BaseActuatorCfg(velocity_limit=0.5),
        "joint_gripper": BaseActuatorCfg(velocity_limit=0.5),
        "joint_finger_right": BaseActuatorCfg(velocity_limit=0.5),
        "joint_finger_left": BaseActuatorCfg(velocity_limit=0.5),
        "joint_head_pan": BaseActuatorCfg(velocity_limit=0.5),
        "joint_head_tilt": BaseActuatorCfg(velocity_limit=0.5),
    }
    # Filled 2026-11-28 from the URDF <limit> elements (robot check config row).
    joint_limits: dict[str, tuple[float, float]] = {
        "joint_torso": (-4.49, 1.35),
        "joint_shoulder": (-2.66, 3.18),
        "joint_bicep": (-2.13, 3.71),
        "joint_elbow": (-2.05, 3.79),
        "joint_forearm": (-2.92, 2.92),
        "joint_wrist": (-1.79, 1.79),
        "joint_gripper": (-4.49, 1.35),
        "joint_finger_right": (-0.0001, 1.3),
        "joint_finger_left": (-0.0001, 1.3),
        "joint_head_pan": (-3.79, 2.22),
        "joint_head_tilt": (-1.17, 1.17),
    }
    # Filled 2026-11-28: the zero pose clamped into the limits, which is what the handlers fell back to without a default (robot check config row).
    default_joint_positions: dict[str, float] = {
        "joint_torso": 0.0,
        "joint_shoulder": 0.0,
        "joint_bicep": 0.0,
        "joint_elbow": 0.0,
        "joint_forearm": 0.0,
        "joint_wrist": 0.0,
        "joint_gripper": 0.0,
        "joint_finger_right": 0.0,
        "joint_finger_left": 0.0,
        "joint_head_pan": 0.0,
        "joint_head_tilt": 0.0,
    }
    # ee_body_name: str = "panda_hand"
