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

    ISAACLAB = "isaaclab"
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


