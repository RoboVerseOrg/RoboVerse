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
    ISAACGYM = "isaacgym"
    GENESIS = "genesis"
    PYREP = "pyrep"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    SAPIEN2 = "sapien2"
    SAPIEN3 = "sapien3"
    BLENDER = "blender"
    MJX = "mjx"


class RobotType(enum.Enum):
    """Robot type."""

    FRANKA = "franka"
    IIWA = "iiwa"
    UR5E_ROBOTIQ_2F_85 = "ur5e_robotiq_2f_85"


class StateKey(enum.Enum):
    """State key."""

    POS = "pos"
    ROT = "rot"
    VEL = "vel"
    ANG_VEL = "ang_vel"
    DOF_POS = "dof_pos"
    DOF_VEL = "dof_vel"
    DOF_POS_TARGET = "dof_pos_target"
