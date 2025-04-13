from __future__ import annotations

import gymnasium as gym
import numpy as np

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AllegroHandCfg(BaseRobotCfg):
    name: str = "allegro_hand"
    num_joints: int = 16
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/allegro_hand/usd/allegro_hand.usd"
    mjcf_path: str = "roboverse_data/robots/allegro_hand/mjcf/allegro_hand.xml"
    urdf_path: str = "roboverse_data/robots/allegro_hand/urdf/allegro_touch_sensor.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "index_joint_0": BaseActuatorCfg(velocity_limit=30.0),
        "index_joint_1": BaseActuatorCfg(velocity_limit=30.0),
        "index_joint_2": BaseActuatorCfg(velocity_limit=30.0),
        "index_joint_3": BaseActuatorCfg(velocity_limit=30.0),
        "middle_joint_0": BaseActuatorCfg(velocity_limit=30.0),
        "middle_joint_1": BaseActuatorCfg(velocity_limit=30.0),
        "middle_joint_2": BaseActuatorCfg(velocity_limit=30.0),
        "middle_joint_3": BaseActuatorCfg(velocity_limit=30.0),
        "ring_joint_0": BaseActuatorCfg(velocity_limit=30.0),
        "ring_joint_1": BaseActuatorCfg(velocity_limit=30.0),
        "ring_joint_2": BaseActuatorCfg(velocity_limit=30.0),
        "ring_joint_3": BaseActuatorCfg(velocity_limit=30.0),
        "thumb_joint_0": BaseActuatorCfg(velocity_limit=30.0),
        "thumb_joint_1": BaseActuatorCfg(velocity_limit=30.0),
        "thumb_joint_2": BaseActuatorCfg(velocity_limit=30.0),
        "thumb_joint_3": BaseActuatorCfg(velocity_limit=30.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "index_joint_0": (-0.47, 0.47),
        "index_joint_1": (-0.196, 1.61),
        "index_joint_2": (-0.174, 1.709),
        "index_joint_3": (-0.227, 1.618),
        "middle_joint_0": (-0.47, 0.47),
        "middle_joint_1": (-0.196, 1.61),
        "middle_joint_2": (-0.174, 1.709),
        "middle_joint_3": (-0.227, 1.618),
        "ring_joint_0": (-0.47, 0.47),
        "ring_joint_1": (-0.196, 1.61),
        "ring_joint_2": (-0.174, 1.709),
        "ring_joint_3": (-0.227, 1.618),
        "thumb_joint_0": (0.263, 1.396),
        "thumb_joint_1": (-0.105, 1.163),
        "thumb_joint_2": (-0.189, 1.644),
        "thumb_joint_3": (-0.162, 1.719),
    }

    default_joint_positions: dict[str, float] = {
        "index_joint_0": 0.0,
        "index_joint_1": 0.0,
        "index_joint_2": 0.0,
        "index_joint_3": 0.0,
        "middle_joint_0": 0.0,
        "middle_joint_1": 0.0,
        "middle_joint_2": 0.0,
        "middle_joint_3": 0.0,
        "ring_joint_0": 0.0,
        "ring_joint_1": 0.0,
        "ring_joint_2": 0.0,
        "ring_joint_3": 0.0,
        "thumb_joint_0": 0.0,
        "thumb_joint_1": 0.0,
        "thumb_joint_2": 0.0,
        "thumb_joint_3": 0.0,
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 0.5)

    default_orientation: tuple[float, float, float, float] = (
        0.2575507164001465,
        0.28304457664489746,
        0.6833299994468689,
        -0.6217824220657349,
    )  # w, x, y, z

    _joint_limit_values = list(joint_limits.values())
    _action_low = np.array([limit[0] for limit in _joint_limit_values], dtype=np.float32)
    _action_high = np.array([limit[1] for limit in _joint_limit_values], dtype=np.float32)

    action_space: gym.Space = gym.spaces.Box(low=_action_low, high=_action_high, shape=(16,), dtype=np.float32)
