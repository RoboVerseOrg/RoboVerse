from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class H1SimpleHandCfg(BaseRobotCfg):
    name: str = "h1_simple_hand"
    num_joints: int = 52
    mjcf_path: str = "roboverse_data/robots/h1_simple_hand/mjcf/h1_simple_hand.xml"

    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
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
        "left_shoulder_pitch": BaseActuatorCfg(),
        "left_shoulder_roll": BaseActuatorCfg(),
        "left_shoulder_yaw": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
        "left_hand": BaseActuatorCfg(),
        "right_shoulder_pitch": BaseActuatorCfg(),
        "right_shoulder_roll": BaseActuatorCfg(),
        "right_shoulder_yaw": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
        "right_hand": BaseActuatorCfg(),
        "L_index_proximal": BaseActuatorCfg(),
        "L_index_intermediate": BaseActuatorCfg(),
        "L_middle_proximal": BaseActuatorCfg(),
        "L_middle_intermediate": BaseActuatorCfg(),
        "L_ring_proximal": BaseActuatorCfg(),
        "L_ring_intermediate": BaseActuatorCfg(),
        "L_pinky_proximal": BaseActuatorCfg(),
        "L_pinky_intermediate": BaseActuatorCfg(),
        "L_thumb_proximal_yaw": BaseActuatorCfg(),
        "L_thumb_proximal_pitch": BaseActuatorCfg(),
        "L_thumb_intermediate": BaseActuatorCfg(),
        "L_thumb_distal": BaseActuatorCfg(),
        "R_index_proximal": BaseActuatorCfg(),
        "R_index_intermediate": BaseActuatorCfg(),
        "R_middle_proximal": BaseActuatorCfg(),
        "R_middle_intermediate": BaseActuatorCfg(),
        "R_ring_proximal": BaseActuatorCfg(),
        "R_ring_intermediate": BaseActuatorCfg(),
        "R_pinky_proximal": BaseActuatorCfg(),
        "R_pinky_intermediate": BaseActuatorCfg(),
        "R_thumb_proximal_yaw": BaseActuatorCfg(),
        "R_thumb_proximal_pitch": BaseActuatorCfg(),
        "R_thumb_intermediate": BaseActuatorCfg(),
        "R_thumb_distal": BaseActuatorCfg(),
    }
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
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "left_hand": (-6, 6),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
        "right_hand": (-6, 6),
        "L_index_proximal": (-1, 1),
        "L_index_intermediate": (-1, 1),
        "L_middle_proximal": (-1, 1),
        "L_middle_intermediate": (-1, 1),
        "L_ring_proximal": (-1, 1),
        "L_ring_intermediate": (-1, 1),
        "L_pinky_proximal": (-1, 1),
        "L_pinky_intermediate": (-1, 1),
        "L_thumb_proximal_yaw": (-1, 1),
        "L_thumb_proximal_pitch": (-1, 1),
        "L_thumb_intermediate": (-1, 1),
        "L_thumb_distal": (-1, 1),
        "R_index_proximal": (-1, 1),
        "R_index_intermediate": (-1, 1),
        "R_middle_proximal": (-1, 1),
        "R_middle_intermediate": (-1, 1),
        "R_ring_proximal": (-1, 1),
        "R_ring_intermediate": (-1, 1),
        "R_pinky_proximal": (-1, 1),
        "R_pinky_intermediate": (-1, 1),
        "R_thumb_proximal_yaw": (-1, 1),
        "R_thumb_proximal_pitch": (-1, 1),
        "R_thumb_intermediate": (-1, 1),
        "R_thumb_distal": (-1, 1),
    }
