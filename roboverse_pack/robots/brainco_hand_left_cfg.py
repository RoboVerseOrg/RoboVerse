"""Configuration for BrainCo Hand Left."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class BraincoHandLeftCfg(RobotCfg):
    """Configuration for BrainCo Hand Left.

    Joint structure:
    - 11 revolute joints total
    - 6 independent joints (actively controlled)
    - 5 mimic joints (passively follow independent joints)

    Independent joints:
    - left_thumb_metacarpal_joint: Thumb base rotation
    - left_thumb_proximal_joint: Thumb proximal flexion
    - left_index_proximal_joint: Index finger flexion
    - left_middle_proximal_joint: Middle finger flexion
    - left_ring_proximal_joint: Ring finger flexion
    - left_pinky_proximal_joint: Pinky finger flexion

    Mimic joints (passive):
    - left_thumb_distal_joint: Mimics left_thumb_proximal_joint (multiplier=1.0)
    - left_index_distal_joint: Mimics left_index_proximal_joint (multiplier=1.155)
    - left_middle_distal_joint: Mimics left_middle_proximal_joint (multiplier=1.155)
    - left_ring_distal_joint: Mimics left_ring_proximal_joint (multiplier=1.155)
    - left_pinky_distal_joint: Mimics left_pinky_proximal_joint (multiplier=1.155)
    """

    name: str = "brainco_hand_left"
    num_joints: int = 11  # Total revolute joints
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/brainco_hand/urdf/brainco_left.urdf"
    mjcf_path: str = "roboverse_data/robots/brainco_hand/mjcf/brainco_left.xml"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    # Isaac Gym specific settings
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = False

    # Set initial pose for the hand (vertical orientation)
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.15)  # 15cm above ground
    default_orientation: tuple[float, float, float, float] = (
        0.7071,
        -0.7071,
        0.0,
        0.0,
    )  # (w, x, y, z) - rotated -90° around X axis

    actuators: dict[str, BaseActuatorCfg] = {
        # Thumb (2 actuated + 1 passive = 3 joints)
        "left_thumb_metacarpal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_thumb_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_thumb_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Index finger (1 actuated + 1 passive = 2 joints)
        "left_index_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_index_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Middle finger (1 actuated + 1 passive = 2 joints)
        "left_middle_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_middle_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Ring finger (1 actuated + 1 passive = 2 joints)
        "left_ring_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_ring_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Pinky finger (1 actuated + 1 passive = 2 joints)
        "left_pinky_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "left_pinky_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
    }

    # Joint limits from MJCF (in radians)
    joint_limits: dict[str, tuple[float, float]] = {
        # Thumb
        "left_thumb_metacarpal_joint": (0.0, 1.5184),
        "left_thumb_proximal_joint": (0.0, 1.0472),
        "left_thumb_distal_joint": (0.0, 1.0472),  # Passive, same range as proximal
        # Index finger
        "left_index_proximal_joint": (0.0, 1.4661),
        "left_index_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Middle finger
        "left_middle_proximal_joint": (0.0, 1.4661),
        "left_middle_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Ring finger
        "left_ring_proximal_joint": (0.0, 1.4661),
        "left_ring_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Pinky finger
        "left_pinky_proximal_joint": (0.0, 1.4661),
        "left_pinky_distal_joint": (0.0, 1.693),  # Passive, x 1.155
    }

    default_joint_positions: dict[str, float] = {
        # Thumb - slightly flexed
        "left_thumb_metacarpal_joint": 0.2,
        "left_thumb_proximal_joint": 0.1,
        "left_thumb_distal_joint": 0.1,  # mimic x 1.0
        # Index finger - slightly flexed
        "left_index_proximal_joint": 0.2,
        "left_index_distal_joint": 0.231,  # mimic x 1.155
        # Middle finger - slightly flexed
        "left_middle_proximal_joint": 0.2,
        "left_middle_distal_joint": 0.231,  # mimic x 1.155
        # Ring finger - slightly flexed
        "left_ring_proximal_joint": 0.2,
        "left_ring_distal_joint": 0.231,  # mimic x 1.155
        # Pinky finger - slightly flexed
        "left_pinky_proximal_joint": 0.2,
        "left_pinky_distal_joint": 0.231,  # mimic x 1.155
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "left_thumb_metacarpal_joint": "position",
        "left_thumb_proximal_joint": "position",
        "left_thumb_distal_joint": "position",
        "left_index_proximal_joint": "position",
        "left_index_distal_joint": "position",
        "left_middle_proximal_joint": "position",
        "left_middle_distal_joint": "position",
        "left_ring_proximal_joint": "position",
        "left_ring_distal_joint": "position",
        "left_pinky_proximal_joint": "position",
        "left_pinky_distal_joint": "position",
    }
