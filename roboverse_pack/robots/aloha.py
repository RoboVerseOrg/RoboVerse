from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class AlohaCfg(RobotCfg):
    """Cfg for the ALOHA dual-arm robot.

    Args:
        RobotCfg (_type_): _description_
    """

    name: str = "aloha"
    num_joints: int = 16  # 8 joints per arm (6 arm + 2 gripper)
    fix_base_link: bool = True
    mjx_mjcf_path: str = "roboverse_data/robots/aloha/mjcf/assets/mjx_aloha.xml"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False

    actuators: dict[str, BaseActuatorCfg] = {
        # Left arm joints
        "left/waist": BaseActuatorCfg(stiffness=43, damping=5.76, velocity_limit=3.14158),
        "left/shoulder": BaseActuatorCfg(stiffness=265, damping=20.0, velocity_limit=1.25664),
        "left/elbow": BaseActuatorCfg(stiffness=227, damping=18.49, velocity_limit=1.6057),
        "left/forearm_roll": BaseActuatorCfg(stiffness=78, damping=6.78, velocity_limit=3.14158),
        "left/wrist_angle": BaseActuatorCfg(stiffness=37, damping=6.28, velocity_limit=2.23402),
        "left/wrist_rotate": BaseActuatorCfg(stiffness=10.4, damping=1.2, velocity_limit=3.14158),
        "left/left_finger": BaseActuatorCfg(stiffness=365, damping=40, velocity_limit=0.037, is_ee=True),
        "left/right_finger": BaseActuatorCfg(stiffness=365, damping=40, velocity_limit=0.037, is_ee=True),
        # Right arm joints
        "right/waist": BaseActuatorCfg(stiffness=43, damping=5.76, velocity_limit=3.14158),
        "right/shoulder": BaseActuatorCfg(stiffness=265, damping=20.0, velocity_limit=1.25664),
        "right/elbow": BaseActuatorCfg(stiffness=227, damping=18.49, velocity_limit=1.6057),
        "right/forearm_roll": BaseActuatorCfg(stiffness=78, damping=6.78, velocity_limit=3.14158),
        "right/wrist_angle": BaseActuatorCfg(stiffness=37, damping=6.28, velocity_limit=2.23402),
        "right/wrist_rotate": BaseActuatorCfg(stiffness=10.4, damping=1.2, velocity_limit=3.14158),
        "right/left_finger": BaseActuatorCfg(stiffness=365, damping=40, velocity_limit=0.037, is_ee=True),
        "right/right_finger": BaseActuatorCfg(stiffness=365, damping=40, velocity_limit=0.037, is_ee=True),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Left arm joint limits
        "left/waist": (-3.14158, 3.14158),
        "left/shoulder": (-1.85005, 1.25664),
        "left/elbow": (-1.76278, 1.6057),
        "left/forearm_roll": (-3.14158, 3.14158),
        "left/wrist_angle": (-1.8675, 2.23402),
        "left/wrist_rotate": (-3.14158, 3.14158),
        "left/left_finger": (0.0, 0.041),  # Physical limits: 0cm to 4.1cm
        "left/right_finger": (0.0, 0.041),
        # Right arm joint limits
        "right/waist": (-3.14158, 3.14158),
        "right/shoulder": (-1.85005, 1.25664),
        "right/elbow": (-1.76278, 1.6057),
        "right/forearm_roll": (-3.14158, 3.14158),
        "right/wrist_angle": (-1.8675, 2.23402),
        "right/wrist_rotate": (-3.14158, 3.14158),
        "right/left_finger": (0.0, 0.041),
        "right/right_finger": (0.0, 0.041),
    }

    # End effector configuration - using gripper links as EE bodies

    control_type: dict[str, Literal["position", "effort"]] = {
        # Left arm control types
        "left/waist": "position",
        "left/shoulder": "position",
        "left/elbow": "position",
        "left/forearm_roll": "position",
        "left/wrist_angle": "position",
        "left/wrist_rotate": "position",
        "left/left_finger": "position",
        "left/right_finger": "position",
        # Right arm control types
        "right/waist": "position",
        "right/shoulder": "position",
        "right/elbow": "position",
        "right/forearm_roll": "position",
        "right/wrist_angle": "position",
        "right/wrist_rotate": "position",
        "right/left_finger": "position",
        "right/right_finger": "position",
    }

    # Gripper control - open and close positions
    gripper_open_q = [0.037, 0.037, 0.037, 0.037]  # Both arms open
    gripper_close_q = [0.002, 0.002, 0.002, 0.002]  # Both arms closed
