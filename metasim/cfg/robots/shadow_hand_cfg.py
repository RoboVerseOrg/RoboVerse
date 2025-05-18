from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class ShadowHandCfg(BaseRobotCfg):
    """Cfg for the Shadow Hand robot."""

    name: str = "shadow_hand"
    num_joints: int = 24
    fix_base_link: bool = True
    mjcf_path: str = "roboverse_data/robots/shadow_hand/mjcf/shadow_hand.xml"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "robot0:WRJ1": BaseActuatorCfg(),
        "robot0:WRJ0": BaseActuatorCfg(),
        "robot0:FFJ3": BaseActuatorCfg(),
        "robot0:FFJ2": BaseActuatorCfg(),
        "robot0:FFJ1": BaseActuatorCfg(),
        "robot0:FFJ0": BaseActuatorCfg(actionable=False),
        "robot0:MFJ3": BaseActuatorCfg(),
        "robot0:MFJ2": BaseActuatorCfg(),
        "robot0:MFJ1": BaseActuatorCfg(),
        "robot0:MFJ0": BaseActuatorCfg(actionable=False),
        "robot0:RFJ3": BaseActuatorCfg(),
        "robot0:RFJ2": BaseActuatorCfg(),
        "robot0:RFJ1": BaseActuatorCfg(),
        "robot0:RFJ0": BaseActuatorCfg(actionable=False),
        "robot0:LFJ4": BaseActuatorCfg(),
        "robot0:LFJ3": BaseActuatorCfg(),
        "robot0:LFJ2": BaseActuatorCfg(),
        "robot0:LFJ1": BaseActuatorCfg(),
        "robot0:LFJ0": BaseActuatorCfg(actionable=False),
        "robot0:THJ4": BaseActuatorCfg(),
        "robot0:THJ3": BaseActuatorCfg(),
        "robot0:THJ2": BaseActuatorCfg(),
        "robot0:THJ1": BaseActuatorCfg(),
        "robot0:THJ0": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "robot0:WRJ1": (-0.489, 0.14),
        "robot0:WRJ0": (-0.698, 0.489),
        "robot0:FFJ3": (-0.349, 0.349),
        "robot0:FFJ2": (0, 1.571),
        "robot0:FFJ1": (0, 1.571),
        "robot0:FFJ0": (0, 1.571),
        "robot0:MFJ3": (-0.349, 0.349),
        "robot0:MFJ2": (0, 1.571),
        "robot0:MFJ1": (0, 1.571),
        "robot0:MFJ0": (0, 1.571),
        "robot0:RFJ3": (-0.349, 0.349),
        "robot0:RFJ2": (0, 1.571),
        "robot0:RFJ1": (0, 1.571),
        "robot0:RFJ0": (0, 1.571),
        "robot0:LFJ4": (0, 0.785),
        "robot0:LFJ3": (-0.349, 0.349),
        "robot0:LFJ2": (0, 1.571),
        "robot0:LFJ1": (0, 1.571),
        "robot0:LFJ0": (0, 1.571),
        "robot0:THJ4": (-1.047, 1.047),
        "robot0:THJ3": (0, 1.222),
        "robot0:THJ2": (-0.209, 0.209),
        "robot0:THJ1": (-0.524, 0.524),
        "robot0:THJ0": (-1.571, 0),
    }

    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False
