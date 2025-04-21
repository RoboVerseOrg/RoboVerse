from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AllegroHandCfg(BaseRobotCfg):
    name: str = "allegro_hand"
    num_joints: int = 16
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/allegro_hand/usd/allegro_hand_instanceable.usd"
    mjcf_path: str = "roboverse_data/robots/allegro_hand/mjcf/allegro_hand.xml"
    urdf_path: str = "roboverse_data/robots/allegro_hand/urdf/allegro_touch_sensor.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "index_joint_0": BaseActuatorCfg(velocity_limit=6.283),
        "index_joint_1": BaseActuatorCfg(velocity_limit=6.283),
        "index_joint_2": BaseActuatorCfg(velocity_limit=6.283),
        "index_joint_3": BaseActuatorCfg(velocity_limit=6.283),
        "middle_joint_0": BaseActuatorCfg(velocity_limit=6.283),
        "middle_joint_1": BaseActuatorCfg(velocity_limit=6.283),
        "middle_joint_2": BaseActuatorCfg(velocity_limit=6.283),
        "middle_joint_3": BaseActuatorCfg(velocity_limit=6.283),
        "ring_joint_0": BaseActuatorCfg(velocity_limit=6.283),
        "ring_joint_1": BaseActuatorCfg(velocity_limit=6.283),
        "ring_joint_2": BaseActuatorCfg(velocity_limit=6.283),
        "ring_joint_3": BaseActuatorCfg(velocity_limit=6.283),
        "thumb_joint_0": BaseActuatorCfg(velocity_limit=6.283),
        "thumb_joint_1": BaseActuatorCfg(velocity_limit=6.283),
        "thumb_joint_2": BaseActuatorCfg(velocity_limit=6.283),
        "thumb_joint_3": BaseActuatorCfg(velocity_limit=6.283),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "index_joint_0": (-0.558488888889, 0.558488888889),
        "index_joint_1": (-0.279244444444, 1.727825),
        "index_joint_2": (-0.279244444444, 1.727825),
        "index_joint_3": (-0.279244444444, 1.727825),
        "middle_joint_0": (-0.558488888889, 0.558488888889),
        "middle_joint_1": (-0.279244444444, 1.727825),
        "middle_joint_2": (-0.279244444444, 1.727825),
        "middle_joint_3": (-0.279244444444, 1.727825),
        "ring_joint_0": (-0.558488888889, 0.558488888889),
        "ring_joint_1": (-0.279244444444, 1.727825),
        "ring_joint_2": (-0.279244444444, 1.727825),
        "ring_joint_3": (-0.279244444444, 1.727825),
        "thumb_joint_0": (0.279244444444, 1.57075),
        "thumb_joint_1": (-0.331602777778, 1.15188333333),
        "thumb_joint_2": (-0.279244444444, 1.727825),
        "thumb_joint_3": (-0.279244444444, 1.76273055556),
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
