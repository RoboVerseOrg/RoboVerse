from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class UnitreeDex31LeftCfg(BaseRobotCfg):
    """Cfg for the Unitree Dexterous Hand 3-1 robot."""

    name: str = "unitree_dex3_1_left"

    num_joints: int = 7
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/unitree_dex3-1/urdf/unitree_dex3_left.urdf"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "left_hand_thumb_0_joint": BaseActuatorCfg(),
        "left_hand_thumb_1_joint": BaseActuatorCfg(),
        "left_hand_thumb_2_joint": BaseActuatorCfg(),
        "left_hand_middle_0_joint": BaseActuatorCfg(),
        "left_hand_middle_1_joint": BaseActuatorCfg(),
        "left_hand_index_0_joint": BaseActuatorCfg(),
        "left_hand_index_1_joint": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "left_hand_thumb_0_joint": (-1.04, -1.04),
        "left_hand_thumb_1_joint": (-0.72, 0.92),
        "left_hand_thumb_2_joint": (0.0, 1.74),
        "left_hand_middle_0_joint": (-1.57, 0.0),
        "left_hand_middle_1_joint": (-1.74, 0.0),
        "left_hand_index_0_joint": (-1.57, 0.0),
        "left_hand_index_1_joint": (-1.75, 0.0),
    }

    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False
