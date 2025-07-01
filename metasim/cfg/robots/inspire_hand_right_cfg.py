from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class InspireHandRightCfg(BaseRobotCfg):
    """Cfg for the Inspire Hand robot."""

    name: str = "inspire_hand_right"
    num_joints: int = 14  # + thumb_intermediate_joint + thumb_distal_joint
    fix_base_link: bool = True
    urdf_path = "/home/zihalu/dexretargeting/assets/robots/hands/inspire_hand/inspire_hand_right.urdf"
    isaacgym_read_mjcf: bool = True
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "index_proximal_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "middle_proximal_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "ring_proximal_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "pinky_proximal_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "thumb_proximal_pitch_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "thumb_proximal_yaw_joint": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        # mimic joints
        "index_intermediate_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "middle_intermediate_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "ring_intermediate_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "pinky_intermediate_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "thumb_intermediate_pitch_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "thumb_intermediate_yaw_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "thumb_intermediate_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
        "thumb_distal_joint": BaseActuatorCfg(stiffness=0.0, damping=0.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "index_proximal_joint": (0.0, 1.47),
        "middle_proximal_joint": (0.0, 1.47),
        "ring_proximal_joint": (0.0, 1.47),
        "pinky_proximal_joint": (0.0, 1.47),
        "thumb_proximal_pitch_joint": (0.0, 0.6),
        "thumb_proximal_yaw_joint": (0.0, 1.308),
        "index_intermediate_joint": (-0.04545, 1.56),
        "middle_intermediate_joint": (-0.04545, 1.56),
        "ring_intermediate_joint": (-0.04545, 1.56),
        "pinky_intermediate_joint": (-0.04545, 1.56),
        "thumb_intermediate_pitch_joint": (0.0, 0.8),
        "thumb_intermediate_yaw_joint": (0.0, 1.3),
        "thumb_intermediate_joint": (0.0, 0.8),
        "thumb_distal_joint": (0.0, 0.4),
    }

    default_joint_positions: dict[str, float] = {
        "index_proximal_joint": 0.0,
        "middle_proximal_joint": 0.0,
        "ring_proximal_joint": 0.0,
        "pinky_proximal_joint": 0.0,
        "thumb_proximal_pitch_joint": 0.0,
        "thumb_proximal_yaw_joint": 0.0,
        "index_intermediate_joint": 0.0,
        "middle_intermediate_joint": 0.0,
        "ring_intermediate_joint": 0.0,
        "pinky_intermediate_joint": 0.0,
        "thumb_intermediate_pitch_joint": 0.0,
        "thumb_intermediate_yaw_joint": 0.0,
        "thumb_intermediate_joint": 0.0,
        "thumb_distal_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "index_proximal_joint": "position",
        "middle_proximal_joint": "position",
        "ring_proximal_joint": "position",
        "pinky_proximal_joint": "position",
        "thumb_proximal_pitch_joint": "position",
        "thumb_proximal_yaw_joint": "position",
        "index_intermediate_joint": "position",
        "middle_intermediate_joint": "position",
        "ring_intermediate_joint": "position",
        "pinky_intermediate_joint": "position",
        "thumb_intermediate_pitch_joint": "position",
        "thumb_intermediate_yaw_joint": "position",
        "thumb_intermediate_joint": "position",
        "thumb_distal_joint": "position",
    }

    default_position = [0.0, 0.0, 0.5]
