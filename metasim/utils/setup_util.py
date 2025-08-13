"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import importlib

from loguru import logger as log

from metasim.constants import SimType
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scene import SceneCfg
from metasim.sim.parallel import ParallelSimWrapper
from metasim.utils import is_camel_case, is_snake_case, to_camel_case


def get_sim_handler_class(sim: SimType):
    """Get the simulator handler class from the simulator type.

    Args:
        sim: The type of the simulator.

    Returns:
        The simulator handler class.
    """
    if sim == SimType.ISAACLAB:
        try:
            from metasim.sim.isaaclab import IsaaclabHandler

            return IsaaclabHandler
        except ImportError as e:
            log.error("IsaacLab is not installed, please install it first")
            raise e
    elif sim == SimType.ISAACGYM:
        try:
            from metasim.sim.isaacgym import IsaacgymHandler

            return IsaacgymHandler
        except ImportError as e:
            log.error("IsaacGym is not installed, please install it first")
            raise e
    elif sim == SimType.GENESIS:
        try:
            from metasim.sim.genesis import GenesisHandler

            return GenesisHandler
        except ImportError as e:
            log.error("Genesis is not installed, please install it first")
            raise e
    elif sim == SimType.PYREP:
        try:
            from metasim.sim.pyrep import PyrepHandler

            return PyrepHandler
        except ImportError as e:
            log.error("PyRep is not installed, please install it first")
            raise e
    elif sim == SimType.PYBULLET:
        try:
            from metasim.sim.pybullet import PybulletHandler

            return PybulletHandler
        except ImportError as e:
            log.error("PyBullet is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN2:
        try:
            from metasim.sim.sapien import Sapien2Handler

            return Sapien2Handler
        except ImportError as e:
            log.error("Sapien is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN3:
        try:
            from metasim.sim.sapien import Sapien3Handler

            return Sapien3Handler
        except ImportError as e:
            log.error("Sapien is not installed, please install it first")
            raise e
    elif sim == SimType.MUJOCO:
        try:
            from metasim.sim.mujoco import MujocoHandler

            return ParallelSimWrapper(MujocoHandler)
        except ImportError as e:
            log.error("Mujoco is not installed, please install it first")
            raise e
    elif sim == SimType.BLENDER:
        try:
            from metasim.sim.blender import BlenderHandler

            return BlenderHandler
        except ImportError as e:
            log.error("Blender is not installed, please install it first")
            raise e
    elif sim == SimType.MJX:
        try:
            from metasim.sim.mjx import MJXHandler

            return MJXHandler
        except ImportError as e:
            log.error("MJX is not installed, please install it first")
            raise e
    else:
        raise ValueError(f"Invalid simulator type: {sim}")


def get_robot(robot_name: str) -> RobotCfg:
    """Get the robot cfg instance from the robot name.

    Args:
        robot_name: The name of the robot.

    Returns:
        The robot cfg instance.
    """
    if is_camel_case(robot_name):
        RobotName = robot_name
    elif is_snake_case(robot_name):
        RobotName = to_camel_case(robot_name)
    else:
        raise ValueError(f"Invalid robot name: {robot_name}, should be in either camel case or snake case")
    module = importlib.import_module("roboverse_pack.robots")
    robot_cls = getattr(module, f"{RobotName}Cfg")
    return robot_cls()


def get_scene(scene_name: str) -> SceneCfg:
    """Get the scene cfg instance from the scene name.

    Args:
        scene_name: The name of the scene.

    Returns:
        The scene cfg instance.
    """
    if is_snake_case(scene_name):
        scene_name = to_camel_case(scene_name)
    try:
        module = importlib.import_module("roboverse_pack.scenes")
        scene_cls = getattr(module, f"{scene_name}Cfg")
        return scene_cls()
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Scene {scene_name} not found: {e}") from e
