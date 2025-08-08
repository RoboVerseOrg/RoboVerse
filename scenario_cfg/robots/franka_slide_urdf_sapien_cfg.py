from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class FrankaSlideUrdfSapienCfg(BaseRobotCfg):
    name: str = "franka"
    num_joints: int = 9
    urdf_path: str = "roboverse_data/robots/franka_rlafford/robots/franka_panda_slider_longer_sapien.urdf"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    actuators: dict[str, BaseActuatorCfg] = {
        "z_free_base1": BaseActuatorCfg(velocity_limit=5.0),
        "z_free_base2": BaseActuatorCfg(velocity_limit=5.0),
        "panda_joint1": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(velocity_limit=2.61),
        "panda_finger_joint1": BaseActuatorCfg(velocity_limit=0.2),
        "panda_finger_joint2": BaseActuatorCfg(velocity_limit=0.2),
    }
    ee_body_name: str = "panda_hand"
