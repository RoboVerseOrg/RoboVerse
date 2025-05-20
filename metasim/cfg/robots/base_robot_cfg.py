from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.cfg.objects import ArticulationObjCfg
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
class BaseRobotCfg(ArticulationObjCfg):
    """Base configuration class for robots."""

    num_joints: int = MISSING
    """Number of joints in the robots."""

    actuators: dict[str, BaseActuatorCfg] = {}
    """Actuators in the robots. The keys are the names of the actuators, and the values are the configurations of the actuators. The names should be consistent with the names in the asset file."""

    ee_body_name: str | None = None
    """Name of the end effector body, which should be consistent with the name in the asset file. This is used for end-effector control."""

    fix_base_link: bool = True
    """Whether to fix the base link. Default to True."""

    joint_limits: dict[str, tuple[float, float]] = {}
    """Joint limits of the robots. The keys are the names of the joints, and the values are the limits of the joints. The names should be consistent with the names in the asset file."""

    default_joint_positions: dict[str, float] = {}
    """Default joint positions of the robots. The keys are the names of the joints, and the values are the default positions of the joints. The names should be consistent with the names in the asset file."""

    enabled_gravity: bool = True
    """Whether to enable gravity. Default to True. If False, the robot will not be affected by gravity."""

    enabled_self_collisions: bool = True
    """Whether to enable self collisions. Default to True. If False, the robot will not collide with itself."""

    isaacgym_flip_visual_attachments: bool = True
    """Whether to flip visual attachments when loading the URDF in IsaacGym. Default to True. For more details, see

    - IsaacGym doc: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.flip_visual_attachments"""

    collapse_fixed_joints: bool = False
    """Whether to collapse fixed joints when loading the URDF in IsaacGym or Genesis. Default to False. For more details, see

    - IsaacGym doc: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.collapse_fixed_joints
    - Genesis doc: https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/urdf.html
    """

    ############################################################
    ## For motion planning and retargetting using cuRobo
    ############################################################

    gripper_release_q: list[float] = MISSING
    gripper_actuate_q: list[float] = MISSING
    curobo_ref_cfg_name: str = MISSING
    curobo_tcp_rel_pos: tuple[float, float, float] = MISSING
    curobo_tcp_rel_rot: tuple[float, float, float] = MISSING

    ############################################################
    ## Experimental
    ############################################################

    control_type: dict[
        str, Literal["position", "effort"]
    ] = {}  # TODO support more controltype, for example, velocity control. Note that effort means use manual pd position controller to get torque and set torque using isaacgym API.
    """[WARNING] This is experimental and subject to change.

    Control type for each joint.
    """
