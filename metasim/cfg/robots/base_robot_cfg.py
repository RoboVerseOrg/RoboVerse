from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass


@configclass
class BaseActuatorCfg:
    """Base configuration class for actuators."""

    velocity_limit: float | None = None  # TODO: None means use the default value (USD joint prim value) or no limit?
    is_ee: bool = False
    damping: float | None = None
    stiffness: float | None = None
    torque_limit: float | None = None
    actionable: bool = True
    """Whether the actuator is actionable, i.e. can be driven by a motor.

    Example:
        Most actuators are actionable, but some are not, e.g. the "left_outer_finger_joint" and "right_outer_finger_joint" of the Robotiq 2F-85 gripper.
        See https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/rig_closed_loop_structures.html for more details.
    """


@configclass
class BaseRobotCfg(ArticulationObjCfg):
    """Base configuration class for robots."""

    num_joints: int = MISSING
    actuators: dict[str, BaseActuatorCfg] = {}
    ee_body_name: str | None = None
    fix_base_link: bool = True
    joint_limits: dict[str, tuple[float, float]] = {}
    default_joint_positions: dict[str, float] = {}

    enabled_gravity: bool = True
    """Whether to enable gravity. Default to True."""

    enabled_self_collisions: bool = True
    """Whether to enable self collisions. Default to True."""

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
    """WARNING: This is experimental and subject to change.

    Control type for each joint.
    """
