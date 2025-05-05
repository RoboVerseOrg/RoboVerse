"""This file contains the constants for the MetaSim."""

import enum


class TaskType(enum.IntEnum):
    """Task type."""

    TABLETOP_MANIPULATION = 0
    """Fixed manipulation"""
    MOBILE_MANIPULATION = 1
    """Mobile manipulation"""
    LOCOMOTION = 2
    """Locomotion"""
    NAVIGATION = 3
    """Navigation"""


class BenchmarkType(enum.Enum):
    """Benchmark type."""

    ## Tabletop Manipulation
    MANISKILL = "ManiSkill"
    METAWORLD = "MetaWorld"
    RLBENCH = "RLbench"
    GAPARTNET = "GAPartNet"
    GAPARTMANIP = "GAPartManip"
    ARNOLD = "ARNOLD"
    CALVIN = "CALVIN"
    UNIDOORMANIP = "UniDoorManip"
    GARMENTLAB = "GarmentLab"
    ROBOSUITE = "RoboSuite"
    ROBOCASA = "RoboCasa"
    OPEN6DOR = "Open6DOR"
    GRASPNET = "GraspNet"
    LIBERO = "Libero"
    GPT = "GPT"

    ## Locomotion
    UH1 = "UH1"

    ## RLAfford
    RLAOPENDOOR = "RLAffordOpenDoor"

    ## SimplerEnv
    SIMPLERENVGRASPSINGLEOPENEDCOKECAN = "SimplerEnvGraspSingleOpenedCokeCan"

    ## humanoid
    HUMANOIDBENCH = "HumanoidBench"

    ## Debug
    DEBUG = "Debug"


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
    SAPIEN = "sapien"
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
