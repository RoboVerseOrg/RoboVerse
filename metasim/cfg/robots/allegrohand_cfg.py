from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AllegroHandCfg(BaseRobotCfg):
    name: str = "allegro_hand"
    num_joints: int = 16
    fix_base_link: bool = True
    usd_path: str = "roboverse_data/robots/allegro_hand/urdf/kuka_allegro_touch_sensor.urdf"
    mjcf_path: str = "roboverse_data/robots/allegro_hand/urdf/kuka_allegro_touch_sensor.urdf"
    urdf_path: str = "roboverse_data/robots/allegro_hand/urdf/kuka_allegro_touch_sensor.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        "index_joint_0": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "index_joint_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "index_joint_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "index_joint_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "middle_joint_0": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "middle_joint_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "middle_joint_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "middle_joint_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "ring_joint_0": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "ring_joint_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "ring_joint_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "ring_joint_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "thumb_joint_0": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "thumb_joint_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "thumb_joint_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
        "thumb_joint_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=0.5, stiffness=3.0, damping=0.1),
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

    # Default orientation for AllegroHand - from IsaacGym reference
    default_orientation: tuple[float, float, float, float] = (
        0.2575507164001465,  # w
        0.28304457664489746,  # x
        0.6833299994468689,  # y
        -0.6217824220657349,  # z
    )  # w, x, y, z

    joint_names: list[str] = [
        "index_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_0",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_0",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_0",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
