# roboverse_data/robots/h1/cfg/h1_cfg.py
from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class H1HandHbCfg(BaseRobotCfg):
    # --------------------------------------------------------------------- #
    #  High-level metadata
    # --------------------------------------------------------------------- #
    name: str = "h1_hand_hb"
    num_joints: int = 61  # 21 body + 20 LH + 20 RH
    mjcf_path: str = "roboverse_data/robots/h1_hand_hb/mjcf/mjx_h1_hand.xml"
    mjx_mjcf_path: str = "roboverse_data/robots/h1_hand_hb/mjcf/mjx_h1_hand.xml"

    # global toggles — leave as-is unless you have a reason to change them
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    # --------------------------------------------------------------------- #
    #  Actuator table
    #  – keys must match the `<position name="…">` entries in your MJCF
    #  – empty BaseActuatorCfg() ⇒ use framework defaults (kp, kv, etc.)
    # --------------------------------------------------------------------- #
    actuators: dict[str, BaseActuatorCfg] = {
        # --------------------- leg & torso --------------------------------
        "left_hip_yaw": BaseActuatorCfg(),
        "left_hip_roll": BaseActuatorCfg(),
        "left_hip_pitch": BaseActuatorCfg(),
        "left_knee": BaseActuatorCfg(),
        "left_ankle": BaseActuatorCfg(),
        "right_hip_yaw": BaseActuatorCfg(),
        "right_hip_roll": BaseActuatorCfg(),
        "right_hip_pitch": BaseActuatorCfg(),
        "right_knee": BaseActuatorCfg(),
        "right_ankle": BaseActuatorCfg(),
        "torso": BaseActuatorCfg(),
        # --------------------- shoulder / arm -----------------------------
        "left_shoulder_pitch": BaseActuatorCfg(),
        "left_shoulder_roll": BaseActuatorCfg(),
        "left_shoulder_yaw": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
        "left_wrist_yaw": BaseActuatorCfg(),
        "right_shoulder_pitch": BaseActuatorCfg(),
        "right_shoulder_roll": BaseActuatorCfg(),
        "right_shoulder_yaw": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
        "right_wrist_yaw": BaseActuatorCfg(),
        # --------------------- left hand (prefix lh_A_) -------------------
        "lh_A_WRJ2": BaseActuatorCfg(),
        "lh_A_WRJ1": BaseActuatorCfg(),
        "lh_A_THJ5": BaseActuatorCfg(),
        "lh_A_THJ4": BaseActuatorCfg(),
        "lh_A_THJ3": BaseActuatorCfg(),
        "lh_A_THJ2": BaseActuatorCfg(),
        "lh_A_THJ1": BaseActuatorCfg(),
        "lh_A_FFJ4": BaseActuatorCfg(),
        "lh_A_FFJ3": BaseActuatorCfg(),
        "lh_A_FFJ0": BaseActuatorCfg(),  # tendon (FFJ2 + FFJ1)
        "lh_A_MFJ4": BaseActuatorCfg(),
        "lh_A_MFJ3": BaseActuatorCfg(),
        "lh_A_MFJ0": BaseActuatorCfg(),  # tendon (MFJ2 + MFJ1)
        "lh_A_RFJ4": BaseActuatorCfg(),
        "lh_A_RFJ3": BaseActuatorCfg(),
        "lh_A_RFJ0": BaseActuatorCfg(),  # tendon (RFJ2 + RFJ1)
        "lh_A_LFJ5": BaseActuatorCfg(),
        "lh_A_LFJ4": BaseActuatorCfg(),
        "lh_A_LFJ3": BaseActuatorCfg(),
        "lh_A_LFJ0": BaseActuatorCfg(),  # tendon (LFJ2 + LFJ1)
        # --------------------- right hand (prefix rh_A_) ------------------
        "rh_A_WRJ2": BaseActuatorCfg(),
        "rh_A_WRJ1": BaseActuatorCfg(),
        "rh_A_THJ5": BaseActuatorCfg(),
        "rh_A_THJ4": BaseActuatorCfg(),
        "rh_A_THJ3": BaseActuatorCfg(),
        "rh_A_THJ2": BaseActuatorCfg(),
        "rh_A_THJ1": BaseActuatorCfg(),
        "rh_A_FFJ4": BaseActuatorCfg(),
        "rh_A_FFJ3": BaseActuatorCfg(),
        "rh_A_FFJ0": BaseActuatorCfg(),
        "rh_A_MFJ4": BaseActuatorCfg(),
        "rh_A_MFJ3": BaseActuatorCfg(),
        "rh_A_MFJ0": BaseActuatorCfg(),
        "rh_A_RFJ4": BaseActuatorCfg(),
        "rh_A_RFJ3": BaseActuatorCfg(),
        "rh_A_RFJ0": BaseActuatorCfg(),
        "rh_A_LFJ5": BaseActuatorCfg(),
        "rh_A_LFJ4": BaseActuatorCfg(),
        "rh_A_LFJ3": BaseActuatorCfg(),
        "rh_A_LFJ0": BaseActuatorCfg(),
    }

    # --------------------------------------------------------------------- #
    #  Soft joint limits (radians) – only for the 21 body DOFs, same as v1
    # --------------------------------------------------------------------- #
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.30, 4.45),
        "left_elbow": (-1.25, 2.61),
        "left_wrist_yaw": (-0.15, 1.57),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.30),
        "right_elbow": (-1.25, 2.61),
        "right_wrist_yaw": (-0.15, 1.57),
    }
