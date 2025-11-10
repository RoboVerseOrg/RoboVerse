from __future__ import annotations

from typing import Literal

from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass


@configclass
class BaseActuatorCfg:
    """Base configuration class for actuators."""

    velocity_limit: float | None = None
    """Velocity limit of the actuator. If not specified, use the value specified in the asset file and interpreted by the simulator."""

    torque_limit: float | None = None
    """Torque limit of the actuator. If not specified, use the value specified in the asset file and interpreted by the simulator."""

    damping: float | None = None
    """Damping of the actuator. If not specified, use the value specified in the asset file and interpreted by the simulator."""

    stiffness: float | None = None
    """Stiffness of the actuator. If not specified, use the value specified in the asset file and interpreted by the simulator."""

    fully_actuated: bool = True
    """Whether the actuator is fully actuated. Default to True.

    Example:
        Most actuators are fully actuated. Otherwise, they are underactuated, e.g. the "left_outer_finger_joint" and "right_outer_finger_joint" of the Robotiq 2F-85 gripper. See https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/rig_closed_loop_structures.html for more details.
    """

    ############################################################
    ## For motion planning and retargetting using cuRobo
    ############################################################

    is_ee: bool = False
    """Whether the actuator is an end effector. Default to False. If True, the actuator will be treated as a part of the end effector for motion planning and retargetting. This configuration may not be used for other purposes."""


@configclass
class RobotCfg(ArticulationObjCfg):
    """Base configuration class for robots."""

    # ==================== Basic Information ====================
    name: str | None = None
    """Robot name for identification and reference"""

    num_joints: int | None = None
    """Number of robot joints, including all movable joints"""

    # # ==================== Asset File Paths ====================
    # # Do not need to fill in all the paths, only fill in the paths that are required for the specific robot and simulation use case
    # usd_path: str | None = None
    # """USD format robot model file path (for IsaacLab, etc.)"""

    # # ==================== Asset File Paths ====================
    # # Do not need to fill in all the paths, only fill in the paths that are required for the specific robot and simulation use case
    # usd_path: str | None = None
    # """USD format robot model file path (for IsaacLab, etc.)"""

    # mjcf_path: str | None = None
    # """MJCF format robot model file path (for MuJoCo, etc.)"""

    # mjx_mjcf_path: str | None = None
    # """MJX format robot model file path (for MJX, etc.)"""

    # urdf_path: str | None = None
    # """URDF format robot model file path (for PyBullet, Sapien, etc.)"""

    # ==================== Physical Properties ====================
    # fix_base_link: bool = True
    # """Whether to fix the robot base."""

    # enabled_gravity: bool = True
    # """Whether to enable gravity effects"""

    # ==================== Joint configuration ====================
    joint_limits: dict[str, tuple[float, float]] | None = None
    """Joint angle limits, keys are joint names, values are (min_value, max_value) tuples (in radians)

    Example:
    joint_limits = {
        "joint1": (-3.14, 3.14),    # -π to π
        "joint2": (-1.57, 1.57),    # -π/2 to π/2
        "gripper_joint": (0.0, 0.04) # 0 to 0.04 radians
    }
    """

    default_joint_positions: dict[str, float] | None = None
    """Default joint positions of the robots. The keys are the names of the joints, and the values are the default positions of the joints. The names should be consistent with the names in the asset file."""

    # ==================== Actuator Configuration ====================
    actuators: dict[str, BaseActuatorCfg] | None = None
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

    # ==================== Control Types ====================
    control_type: dict[str, Literal["position", "effort"]] | None = None
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

    # ==================== Simulator-specific configuration ====================
    isaacgym_flip_visual_attachments: bool = True
    """Whether to flip visual attachments when loading the URDF in IsaacGym. Default to True. For more details, see

    - IsaacGym doc: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.flip_visual_attachments"""

    collapse_fixed_joints: bool = False
    """Whether to collapse fixed joints when loading the URDF in IsaacGym or Genesis. Default to False. For more details, see

    - IsaacGym doc: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.collapse_fixed_joints
    - Genesis doc: https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/urdf.html
    """

    enabled_self_collisions: bool = True
    """Whether to enable self collisions. Default to True. If False, the robot will not collide with itself."""

    # ==================== cuRobo Configuration ====================
    curobo_ref_cfg_name: str | None = None
    """Name of the configuration file for cuRobo. This is used for motion planning and retargetting using cuRobo."""

    ############################################################
    ## Gripper specific configuration
    ############################################################

    # ==================== gripper specific configuration ====================
    gripper_joint_name: str | None = None
    """Name of the gripper joint. This is used for motion planning and retargetting using cuRobo."""

    # ==================== gripper open and close configuration ====================
    gripper_open_q: list[float] | None = None
    """Joint positions of the gripper when the gripper is open. This is used for motion planning and retargetting using cuRobo."""

    gripper_close_q: list[float] | None = None
    """Joint positions of the gripper when the gripper is closed. This is used for motion planning and retargetting using cuRobo."""

    # ==================== cuRobo Specific Configuration ====================
    curobo_tcp_rel_pos: tuple[float, float, float] | None = None
    """Relative position of the TCP to the end effector body link. This is used for motion planning and retargetting using cuRobo."""

    curobo_tcp_rel_rot: tuple[float, float, float] | None = None
    """Relative rotation of the TCP to the end effector body link. This is used for motion planning and retargetting using cuRobo."""

    ############################################################
    ## Single Dexterous Hand specific configuration
    ############################################################

    finger_tips_link_names: list[str] | None = None
    """Names of the finger tips links."""

    finger_tips_offset: list[tuple[float, float, float]] | None = None
    """Offsets of the finger tips links. From the root of the finger links to the actual finger tip position."""

    palm_link_name: str | None = None
    """Name of the palm link."""

    palm_offset: tuple[float, float, float] | None = None
    """Offset of the palm link. From the root of the palm link to the actual palm position."""

    wrist_link_name: str | None = None
    """Name of the wrist link."""

    ############################################################
    ## Humanoid specific configuration
    ############################################################
