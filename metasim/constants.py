"""This file contains the constants for the MetaSim."""

import enum


class PhysicStateType(enum.IntEnum):
    """Physic state type."""

    XFORM = 0
    """No gravity, no collision"""
    GEOM = 1
    """No gravity, with collision"""
    RIGIDBODY = 2
    """With gravity, with collision"""


class SimType(enum.Enum):
    """Simulator type."""

    ISAACSIM = "isaacsim"
    ISAACGYM = "isaacgym"
    GENESIS = "genesis"
    PYREP = "pyrep"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    SAPIEN2 = "sapien2"
    SAPIEN3 = "sapien3"
    BLENDER = "blender"
    MJX = "mjx"
    NEWTON = "newton"
    # Deprecated alias. The standalone IsaacLab handler was removed; the
    # IsaacSim handler now uses the ``isaaclab`` Python package directly.
    # Existing call sites that still reference ``SimType.ISAACLAB``
    # (e.g. ``scripts/conversion/convert_traj_v1_to_v2.py``) keep working
    # — the dispatcher routes ISAACLAB → ISAACSIM and emits a
    # DeprecationWarning on first use. Remove once downstream stops
    # referencing this enum value.
    ISAACLAB = "isaaclab"
