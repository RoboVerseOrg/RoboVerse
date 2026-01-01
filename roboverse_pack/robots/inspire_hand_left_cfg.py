from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class InspireHandLeftCfg(RobotCfg):
    """Configuration for Inspire Hand Left.

    The Inspire Hand has 6 DOF (degrees of freedom) with 12 joints:
    - 6 actuated joints (independent control)
    - 6 passive joints (mechanically coupled to actuated joints)

    DOF breakdown:
    - Thumb: 2 DOF (yaw + pitch), 2 passive (intermediate, distal)
    - Index: 1 DOF (proximal), 1 passive (intermediate)
    - Middle: 1 DOF (proximal), 1 passive (intermediate)
    - Ring: 1 DOF (proximal), 1 passive (intermediate)
    - Pinky: 1 DOF (proximal), 1 passive (intermediate)
    """

    name: str = "inspire_hand_left"
    num_joints: int = 12
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/inspire_hand/urdf/inspire_hand_left.urdf"
    mjcf_path: str = "roboverse_data/robots/inspire_hand/mjcf/inspire_hand_left.xml"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    # Isaac Gym specific settings
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = False

    # NOTE: init_state commented out to use MJCF-defined initial pose
    # MuJoCo's equality constraints (mimic joints) require initial state to satisfy constraints
    # Using MJCF-defined pose avoids constraint violation and BADQACC errors
    # init_state: dict = {
    #     "pos": (0.0, 0.0, 0.15),
    #     "rot": (0.7071, 0.7071, 0.0, 0.0),
    # }

    actuators: dict[str, BaseActuatorCfg] = {
        # Thumb (2 actuated + 2 passive = 4 joints)
        "L_thumb_proximal_yaw_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_thumb_proximal_pitch_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_thumb_intermediate_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        "L_thumb_distal_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Index finger (1 actuated + 1 passive = 2 joints)
        "L_index_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_index_intermediate_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Middle finger (1 actuated + 1 passive = 2 joints)
        "L_middle_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_middle_intermediate_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Ring finger (1 actuated + 1 passive = 2 joints)
        "L_ring_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_ring_intermediate_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
        # Pinky finger (1 actuated + 1 passive = 2 joints)
        "L_pinky_proximal_joint": BaseActuatorCfg(stiffness=10.0, damping=2.0),
        "L_pinky_intermediate_joint": BaseActuatorCfg(fully_actuated=False, effort_limit_sim=0.0),  # Passive/coupled
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Thumb
        "L_thumb_proximal_yaw_joint": (-0.1, 1.3),
        "L_thumb_proximal_pitch_joint": (0.0, 0.5),
        "L_thumb_intermediate_joint": (0.0, 0.8),
        "L_thumb_distal_joint": (0.0, 1.2),
        # Index finger
        "L_index_proximal_joint": (0.0, 1.7),
        "L_index_intermediate_joint": (0.0, 1.7),
        # Middle finger
        "L_middle_proximal_joint": (0.0, 1.7),
        "L_middle_intermediate_joint": (0.0, 1.7),
        # Ring finger
        "L_ring_proximal_joint": (0.0, 1.7),
        "L_ring_intermediate_joint": (0.0, 1.7),
        # Pinky finger
        "L_pinky_proximal_joint": (0.0, 1.7),
        "L_pinky_intermediate_joint": (0.0, 1.7),
    }

    default_joint_positions: dict[str, float] = {
        # Thumb - slightly abducted and flexed
        "L_thumb_proximal_yaw_joint": 0.3,
        "L_thumb_proximal_pitch_joint": 0.1,
        "L_thumb_intermediate_joint": 0.16,  # mimic * 1.6
        "L_thumb_distal_joint": 0.24,  # mimic * 2.4
        # Index finger - slightly flexed
        "L_index_proximal_joint": 0.2,
        "L_index_intermediate_joint": 0.2,  # mimic * 1.0
        # Middle finger - slightly flexed
        "L_middle_proximal_joint": 0.2,
        "L_middle_intermediate_joint": 0.2,  # mimic * 1.0
        # Ring finger - slightly flexed
        "L_ring_proximal_joint": 0.2,
        "L_ring_intermediate_joint": 0.2,  # mimic * 1.0
        # Pinky finger - slightly flexed
        "L_pinky_proximal_joint": 0.2,
        "L_pinky_intermediate_joint": 0.2,  # mimic * 1.0
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "L_thumb_proximal_yaw_joint": "position",
        "L_thumb_proximal_pitch_joint": "position",
        "L_thumb_intermediate_joint": "position",
        "L_thumb_distal_joint": "position",
        "L_index_proximal_joint": "position",
        "L_index_intermediate_joint": "position",
        "L_middle_proximal_joint": "position",
        "L_middle_intermediate_joint": "position",
        "L_ring_proximal_joint": "position",
        "L_ring_intermediate_joint": "position",
        "L_pinky_proximal_joint": "position",
        "L_pinky_intermediate_joint": "position",
    }
