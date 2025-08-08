from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class RobotTemplateCfg(BaseRobotCfg):
    """This is a minimal template for creating new robot configurations."""

    # ==================== Basic Information ====================
    name: str = "robot_template"
    """Robot name for identification and reference"""

    num_joints: int = 0
    """Number of robot joints, including all movable joints"""

    # ==================== Asset File Paths ====================
    # Do not need to fill in all the paths, only fill in the paths that are required for the specific robot and simulation use case
    usd_path: str = "roboverse_data/robots/your_robot/usd/your_robot.usd"
    """USD format robot model file path (for IsaacGym, etc.)"""

    mjcf_path: str = "roboverse_data/robots/your_robot/mjcf/your_robot.xml"
    """MJCF format robot model file path (for MuJoCo, etc.)"""

    mjx_mjcf_path: str = "roboverse_data/robots/your_robot/mjcf/mjx_your_robot.xml"
    """MJX format robot model file path (for MJX, etc.)"""

    urdf_path: str = "roboverse_data/robots/your_robot/urdf/your_robot.urdf"
    """URDF format robot model file path (for PyBullet, Sapien, etc.)"""

    # ==================== Physical Properties ====================
    fix_base_link: bool = True
    """Whether to fix the robot base."""

    enabled_gravity: bool = True
    """Whether to enable gravity effects"""

    # Other customizable fields:......

    # ==================== Actuator Configuration ====================
    actuators: dict[str, BaseActuatorCfg] = None
    """Actuator configuration dictionary, keys are joint names, values are actuator configuration objects

    Example:
    actuators = {
        "joint1": BaseActuatorCfg(
            velocity_limit=2.0,      # Velocity limit (rad/s)
            torque_limit=100.0,      # Torque limit (N⋅m)
            stiffness=1000.0,        # Stiffness coefficient
            damping=100.0,           # Damping coefficient
            fully_actuated=True,     # Whether fully actuated
            is_ee=False              # Whether it's an end effector
        ),
        "gripper_joint": BaseActuatorCfg(
            velocity_limit=0.2,
            torque_limit=10.0,
            stiffness=1000.0,
            damping=100.0,
            is_ee=True  # Mark as end effector
        )
    }
    """

    # ==================== Joint Limits ====================
    joint_limits: dict[str, tuple[float, float]] = None
    """Joint angle limits, keys are joint names, values are (min_value, max_value) tuples (in radians)

    Example:
    joint_limits = {
        "joint1": (-3.14, 3.14),    # -π to π
        "joint2": (-1.57, 1.57),    # -π/2 to π/2
        "gripper_joint": (0.0, 0.04) # 0 to 0.04 radians
    }
    """

    # ==================== Control Types ====================
    control_type: dict[str, Literal["position", "effort"]] = None
    """Control types, keys are joint names, values are control methods

    - "position": Position control
    - "effort": Torque control

    Example:
    control_type = {
        "joint1": "position",
        "joint2": "effort",
        "gripper_joint": "position"
    }
    """


# ==================== Usage Examples ====================
"""
Steps to create a new robot configuration using this template:

1. Copy this template file and rename it
2. Modify the class name and name attribute
3. Set the correct num_joints
4. Update asset file paths
5. Configure actuator parameters
6. Set joint limits and default positions
7. Configure control types
8. Add special configurations as needed

"""
