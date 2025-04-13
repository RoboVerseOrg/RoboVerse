from __future__ import annotations

from dataclasses import MISSING

from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass


@configclass
class BaseActuatorCfg:
    velocity_limit: float | None = None  # TODO: None means use the default value (USD joint prim value) or no limit?
    is_ee: bool = False
    damping: float = 40.0
    stiffness: float = 400.0


@configclass
class BaseRobotCfg(ArticulationObjCfg):
    """Base Cfg class for robots."""

    # Articulation
    num_joints: int = MISSING
    actuators: dict[str, BaseActuatorCfg] = {}
    ee_body_name: str | None = None
    fix_base_link: bool = True
    joint_limits: dict[str, tuple[float, float]] = {}
    default_joint_positions: dict[str, float] = {}
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w
    """
    Joint limits in the format of `{joint_name: (lower_limit, upper_limit)}`.
    Note that different simulators may have different order of joints, so you should not use the order in this dict!
    """

    gripper_release_q: list[float] = MISSING
    gripper_actuate_q: list[float] = MISSING

    # cuRobo Configs
    curobo_ref_cfg_name: str = MISSING
    curobo_tcp_rel_pos: tuple[float, float, float] = MISSING
    curobo_tcp_rel_rot: tuple[float, float, float] = MISSING

    # Simulation
    enabled_gravity: bool = True
    """Whether to enable gravity in the simulation."""
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = True
    collapse_fixed_joints: bool = False
