"""ANYmal quadruped robot configuration."""

from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AnymalCfg(BaseRobotCfg):
    name: str = "anymal"
    num_joints: int = 12
    fix_base_link: bool = False
    scale: list[float] = [1.0, 1.0, 1.0]
    usd_path: str = "roboverse_data/assets/isaacgymenvs/anymal_c/urdf/anymal.urdf"  # Using URDF for now
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/anymal_c/urdf/anymal.urdf"  # Using URDF for now
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/anymal_c/urdf/anymal.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = True
    isaacgym_read_mjcf: bool = False
    collapse_fixed_joints: bool = True

    # Define actuators for each joint
    actuators: dict[str, BaseActuatorCfg] = {
        # Hip Abduction/Adduction
        "LF_HAA": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "LH_HAA": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RF_HAA": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RH_HAA": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        # Hip Flexion/Extension
        "LF_HFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "LH_HFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RF_HFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RH_HFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        # Knee Flexion/Extension
        "LF_KFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "LH_KFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RF_KFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
        "RH_KFE": BaseActuatorCfg(velocity_limit=30.0, torque_limit=40.0, stiffness=85.0, damping=2.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # HAA (Hip Abduction/Adduction) joints
        "LF_HAA": (-0.7, 0.7),
        "LH_HAA": (-0.7, 0.7),
        "RF_HAA": (-0.7, 0.7),
        "RH_HAA": (-0.7, 0.7),
        # HFE (Hip Flexion/Extension) joints
        "LF_HFE": (-1.0, 2.9),
        "LH_HFE": (-2.9, 1.0),
        "RF_HFE": (-1.0, 2.9),
        "RH_HFE": (-2.9, 1.0),
        # KFE (Knee Flexion/Extension) joints
        "LF_KFE": (-2.8, -0.1),
        "LH_KFE": (0.1, 2.8),
        "RF_KFE": (-2.8, -0.1),
        "RH_KFE": (0.1, 2.8),
    }

    # Default joint angles from IsaacGym config
    default_joint_positions: dict[str, float] = {
        "LF_HAA": 0.03,
        "LH_HAA": 0.03,
        "RF_HAA": -0.03,
        "RH_HAA": -0.03,
        "LF_HFE": 0.4,
        "LH_HFE": -0.4,
        "RF_HFE": 0.4,
        "RH_HFE": -0.4,
        "LF_KFE": -0.8,
        "LH_KFE": 0.8,
        "RF_KFE": -0.8,
        "RH_KFE": 0.8,
    }

    # Control parameters
    control_frequency_inv: int = 1  # 60 Hz from IsaacGym config
    action_scale: float = 0.5

    # Default base position
    default_position: list[float] = [0.0, 0.0, 0.62]
    default_orientation: list[float] = [0.0, 0.0, 0.0, 1.0]  # quaternion

    # Observation configuration
    observe_base_position: bool = True
    observe_base_velocity: bool = True
    observe_joint_positions: bool = True
    observe_joint_velocities: bool = True

    # Link names for contact detection
    foot_link_names: list[str] = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
    knee_link_names: list[str] = ["LF_THIGH", "LH_THIGH", "RF_THIGH", "RH_THIGH"]
    base_link_name: str = "base"
