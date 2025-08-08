from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class GoogleRobotStaticCfg(BaseRobotCfg):
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
    # ee_body_name: str = "panda_hand"
