# /home/haoran/lucas/Video2Dex/gym/scenario_cfg/robots/xhand_cfg.py
from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class XhandRightCfg(RobotCfg):
    """Config for the XHand right dexterous hand (12-DoF)."""

    # -------- Basics --------
    name: str = "xhand_right"
    num_joints: int = 12

    # Free base so that we can replay 6-DoF trajectories of the palm.
    fix_base_link: bool = False

    urdf_path: str = "roboverse_data/robots/xhand_right/urdf/xhand_right.urdf"

    # With a free base, disabling gravity is often safer for open-loop replay.
    enabled_gravity: bool = False

    # Self-collision can be toggled as needed.
    enabled_self_collisions: bool = True

    isaacgym_flip_visual_attachments: bool = False

    # Initial base pose (only used as a starting value for a free base).
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.25)
    # Quaternion format: (w, x, y, z)
    default_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Explicit EE link name to avoid name differences across builds.
    ee_link_name: str = "right_hand_ee_link"

    # -------- Joint list (12 DoF) --------
    joint_names: list[str] = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    # -------- Limits (from URDF) --------
    joint_limits: dict[str, tuple[float, float]] = {
        "right_hand_thumb_bend_joint": (0.0, 1.832),
        "right_hand_thumb_rota_joint1": (-0.698, 1.57),
        "right_hand_thumb_rota_joint2": (0.0, 1.57),
        "right_hand_index_bend_joint": (-0.174, 0.174),
        "right_hand_index_joint1": (0.0, 1.919),
        "right_hand_index_joint2": (0.0, 1.919),
        "right_hand_mid_joint1": (0.0, 1.919),
        "right_hand_mid_joint2": (0.0, 1.919),
        "right_hand_ring_joint1": (0.0, 1.919),
        "right_hand_ring_joint2": (0.0, 1.919),
        "right_hand_pinky_joint1": (0.0, 1.919),
        "right_hand_pinky_joint2": (0.0, 1.919),
    }

    # -------- Actuators --------
    # Rough effort/velocity split: proximal (effort≈1.1, vel≈8.63) / distal (effort≈0.4, vel≈14.38).
    actuators: dict[str, BaseActuatorCfg] = {
        # Thumb
        "right_hand_thumb_bend_joint": BaseActuatorCfg(
            velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1
        ),
        "right_hand_thumb_rota_joint1": BaseActuatorCfg(
            velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1
        ),
        "right_hand_thumb_rota_joint2": BaseActuatorCfg(
            velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1
        ),
        # Index
        "right_hand_index_bend_joint": BaseActuatorCfg(
            velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1
        ),
        "right_hand_index_joint1": BaseActuatorCfg(velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1),
        "right_hand_index_joint2": BaseActuatorCfg(velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1),
        # Middle
        "right_hand_mid_joint1": BaseActuatorCfg(velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1),
        "right_hand_mid_joint2": BaseActuatorCfg(velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1),
        # Ring
        "right_hand_ring_joint1": BaseActuatorCfg(velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1),
        "right_hand_ring_joint2": BaseActuatorCfg(velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1),
        # Pinky
        "right_hand_pinky_joint1": BaseActuatorCfg(velocity_limit=8.63, torque_limit=1.1, stiffness=3.0, damping=0.1),
        "right_hand_pinky_joint2": BaseActuatorCfg(velocity_limit=14.38, torque_limit=0.4, stiffness=3.0, damping=0.1),
    }

    # -------- Defaults --------
    default_joint_positions: dict[str, float] = {
        "right_hand_thumb_bend_joint": 0.0,
        "right_hand_thumb_rota_joint1": 0.0,
        "right_hand_thumb_rota_joint2": 0.0,
        "right_hand_index_bend_joint": 0.0,
        "right_hand_index_joint1": 0.0,
        "right_hand_index_joint2": 0.0,
        "right_hand_mid_joint1": 0.0,
        "right_hand_mid_joint2": 0.0,
        "right_hand_ring_joint1": 0.0,
        "right_hand_ring_joint2": 0.0,
        "right_hand_pinky_joint1": 0.0,
        "right_hand_pinky_joint2": 0.0,
    }

    # -------- Control type --------
    control_type: dict[str, Literal["position", "effort"]] = {
        j: "position"
        for j in [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]
    }
