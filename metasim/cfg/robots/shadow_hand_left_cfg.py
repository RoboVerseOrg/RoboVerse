from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class ShadowHandLeftCfg(BaseRobotCfg):
    """Cfg for the Shadow Hand robot."""

    name: str = "shadow_hand_left"
    num_joints: int = 24
    fix_base_link: bool = True
    mjcf_path: str = "roboverse_data/robots/shadow_hand/mjcf/shadow_hand_left.xml"
    usd_path: str = "roboverse_data/robots/shadow_hand/usd/shadow_hand_instanceable.usd"
    isaacgym_read_mjcf: bool = True
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "robot1_WRJ1": BaseActuatorCfg(stiffness=5.0, damping=0.5),
        "robot1_WRJ0": BaseActuatorCfg(stiffness=5.0, damping=0.5),
        "robot1_FFJ3": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_FFJ2": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_FFJ1": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_FFJ0": BaseActuatorCfg(stiffness=1.0, damping=0.1, fully_actuated=False),
        "robot1_MFJ3": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_MFJ2": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_MFJ1": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_MFJ0": BaseActuatorCfg(stiffness=1.0, damping=0.1, fully_actuated=False),
        "robot1_RFJ3": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_RFJ2": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_RFJ1": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_RFJ0": BaseActuatorCfg(stiffness=1.0, damping=0.1, fully_actuated=False),
        "robot1_LFJ4": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_LFJ3": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_LFJ2": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_LFJ1": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_LFJ0": BaseActuatorCfg(stiffness=1.0, damping=0.1, fully_actuated=False),
        "robot1_THJ4": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_THJ3": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_THJ2": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_THJ1": BaseActuatorCfg(stiffness=1.0, damping=0.1),
        "robot1_THJ0": BaseActuatorCfg(stiffness=1.0, damping=0.1),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "robot1_WRJ1": (-0.489, 0.14),
        "robot1_WRJ0": (-0.698, 0.489),
        "robot1_FFJ3": (-0.349, 0.349),
        "robot1_FFJ2": (0, 1.571),
        "robot1_FFJ1": (0, 1.571),
        "robot1_FFJ0": (0, 1.571),
        "robot1_MFJ3": (-0.349, 0.349),
        "robot1_MFJ2": (0, 1.571),
        "robot1_MFJ1": (0, 1.571),
        "robot1_MFJ0": (0, 1.571),
        "robot1_RFJ3": (-0.349, 0.349),
        "robot1_RFJ2": (0, 1.571),
        "robot1_RFJ1": (0, 1.571),
        "robot1_RFJ0": (0, 1.571),
        "robot1_LFJ4": (0, 0.785),
        "robot1_LFJ3": (-0.349, 0.349),
        "robot1_LFJ2": (0, 1.571),
        "robot1_LFJ1": (0, 1.571),
        "robot1_LFJ0": (0, 1.571),
        "robot1_THJ4": (-1.047, 1.047),
        "robot1_THJ3": (0, 1.222),
        "robot1_THJ2": (-0.209, 0.209),
        "robot1_THJ1": (-0.524, 0.524),
        "robot1_THJ0": (-1.571, 0),
    }

    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False

    default_joint_positions: dict[str, float] = {
        "robot1_WRJ1": 0.0,
        "robot1_WRJ0": 0.0,
        "robot1_FFJ3": 0.0,
        "robot1_FFJ2": 0.0,
        "robot1_FFJ1": 0.0,
        "robot1_FFJ0": 0.0,
        "robot1_MFJ3": 0.0,
        "robot1_MFJ2": 0.0,
        "robot1_MFJ1": 0.0,
        "robot1_MFJ0": 0.0,
        "robot1_RFJ3": 0.0,
        "robot1_RFJ2": 0.0,
        "robot1_RFJ1": 0.0,
        "robot1_RFJ0": 0.0,
        "robot1_LFJ4": 0.0,
        "robot1_LFJ3": 0.0,
        "robot1_LFJ2": 0.0,
        "robot1_LFJ1": 0.0,
        "robot1_LFJ0": 0.0,
        "robot1_THJ4": 0.0,
        "robot1_THJ3": 0.0,
        "robot1_THJ2": 0.0,
        "robot1_THJ1": 0.0,
        "robot1_THJ0": 0.0,
    }
    control_type: dict[str, Literal["position", "effort"]] = {
        "robot1_WRJ1": "position",
        "robot1_WRJ0": "position",
        "robot1_FFJ3": "position",
        "robot1_FFJ2": "position",
        "robot1_FFJ1": "position",
        "robot1_FFJ0": "position",
        "robot1_MFJ3": "position",
        "robot1_MFJ2": "position",
        "robot1_MFJ1": "position",
        "robot1_MFJ0": "position",
        "robot1_RFJ3": "position",
        "robot1_RFJ2": "position",
        "robot1_RFJ1": "position",
        "robot1_RFJ0": "position",
        "robot1_LFJ4": "position",
        "robot1_LFJ3": "position",
        "robot1_LFJ2": "position",
        "robot1_LFJ1": "position",
        "robot1_LFJ0": "position",
        "robot1_THJ4": "position",
        "robot1_THJ3": "position",
        "robot1_THJ2": "position",
        "robot1_THJ1": "position",
        "robot1_THJ0": "position",
    }
    default_position = [0.0, 0.0, 0.5]
