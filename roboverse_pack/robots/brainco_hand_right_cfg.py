"""Configuration for BrainCo Hand Right."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class BraincoHandRightCfg(RobotCfg):
    """Configuration for BrainCo Hand Right.

    Joint structure:
    - 11 revolute joints total
    - 6 independent joints (actively controlled)
    - 5 mimic joints (passively follow independent joints)

    Independent joints:
    - right_thumb_metacarpal_joint: Thumb base rotation
    - right_thumb_proximal_joint: Thumb proximal flexion
    - right_index_proximal_joint: Index finger flexion
    - right_middle_proximal_joint: Middle finger flexion
    - right_ring_proximal_joint: Ring finger flexion
    - right_pinky_proximal_joint: Pinky finger flexion

    Mimic joints (passive):
    - right_thumb_distal_joint: Mimics right_thumb_proximal_joint (multiplier=1.0)
    - right_index_distal_joint: Mimics right_index_proximal_joint (multiplier=1.155)
    - right_middle_distal_joint: Mimics right_middle_proximal_joint (multiplier=1.155)
    - right_ring_distal_joint: Mimics right_ring_proximal_joint (multiplier=1.155)
    - right_pinky_distal_joint: Mimics right_pinky_proximal_joint (multiplier=1.155)
    """

    name: str = "brainco_hand_right"
    num_joints: int = 11  # Total revolute joints
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/brainco_hand/urdf/brainco_right.urdf"
    mjcf_path: str = "roboverse_data/robots/brainco_hand/mjcf/brainco_right.xml"
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
        "right_thumb_metacarpal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_thumb_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_thumb_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Index finger (1 actuated + 1 passive = 2 joints)
        "right_index_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_index_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Middle finger (1 actuated + 1 passive = 2 joints)
        "right_middle_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_middle_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Ring finger (1 actuated + 1 passive = 2 joints)
        "right_ring_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_ring_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Pinky finger (1 actuated + 1 passive = 2 joints)
        "right_pinky_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=1.0),
        "right_pinky_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
    }

    # Joint limits from MJCF (in radians)
    joint_limits: dict[str, tuple[float, float]] = {
        # Thumb
        "right_thumb_metacarpal_joint": (0.0, 1.5184),
        "right_thumb_proximal_joint": (0.0, 1.0472),
        "right_thumb_distal_joint": (0.0, 1.0472),  # Passive, same range as proximal
        # Index finger
        "right_index_proximal_joint": (0.0, 1.4661),
        "right_index_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Middle finger
        "right_middle_proximal_joint": (0.0, 1.4661),
        "right_middle_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Ring finger
        "right_ring_proximal_joint": (0.0, 1.4661),
        "right_ring_distal_joint": (0.0, 1.693),  # Passive, x 1.155
        # Pinky finger
        "right_pinky_proximal_joint": (0.0, 1.4661),
        "right_pinky_distal_joint": (0.0, 1.693),  # Passive, x 1.155
    }

    default_joint_positions: dict[str, float] = {
        # Thumb - slightly flexed
        "right_thumb_metacarpal_joint": 0.2,
        "right_thumb_proximal_joint": 0.1,
        "right_thumb_distal_joint": 0.1,  # mimic x 1.0
        # Index finger - slightly flexed
        "right_index_proximal_joint": 0.2,
        "right_index_distal_joint": 0.231,  # mimic x 1.155
        # Middle finger - slightly flexed
        "right_middle_proximal_joint": 0.2,
        "right_middle_distal_joint": 0.231,  # mimic x 1.155
        # Ring finger - slightly flexed
        "right_ring_proximal_joint": 0.2,
        "right_ring_distal_joint": 0.231,  # mimic x 1.155
        # Pinky finger - slightly flexed
        "right_pinky_proximal_joint": 0.2,
        "right_pinky_distal_joint": 0.231,  # mimic x 1.155
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "right_thumb_metacarpal_joint": "position",
        "right_thumb_proximal_joint": "position",
        "right_thumb_distal_joint": "position",
        "right_index_proximal_joint": "position",
        "right_index_distal_joint": "position",
        "right_middle_proximal_joint": "position",
        "right_middle_distal_joint": "position",
        "right_ring_proximal_joint": "position",
        "right_ring_distal_joint": "position",
        "right_pinky_proximal_joint": "position",
        "right_pinky_distal_joint": "position",
    }
