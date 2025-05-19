"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import importlib

import gymnasium as gym
from loguru import logger as log

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenes import SceneCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import SimType
from metasim.utils import is_camel_case, is_snake_case, to_camel_case, to_snake_case


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
    elif sim == SimType.SAPIEN:
        try:
            from metasim.sim.sapien import SapienHandler

            return SapienHandler
        except ImportError as e:
            log.error("Sapien is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN2:
        try:
            from metasim.sim.sapien import SapienHandler

            return SapienHandler
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

            return MujocoHandler
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


def get_sim_env_class(sim: SimType):
    """Get the simulator environment class from the simulator type.

    Args:
        sim: The type of the simulator.

    Returns:
        The simulator environment environment class.
    """
    if sim == SimType.ISAACLAB:
        try:
            from metasim.sim.isaaclab import IsaaclabEnv

            return IsaaclabEnv
        except ImportError as e:
            log.error("IsaacLab is not installed, please install it first")
            raise e
    elif sim == SimType.ISAACGYM:
        try:
            from metasim.sim.isaacgym.isaacgym import IsaacgymEnv

            return IsaacgymEnv
        except ImportError as e:
            log.error("IsaacGym is not installed, please install it first")
            raise e
    elif sim == SimType.GENESIS:
        try:
            from metasim.sim.genesis import GenesisEnv

            return GenesisEnv
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
            from metasim.sim.pybullet import PybulletEnv

            return PybulletEnv
        except ImportError as e:
            log.error("PyBullet is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN or sim == SimType.SAPIEN2:
        try:
            from metasim.sim.sapien import SapienEnv

            return SapienEnv
        except ImportError as e:
            log.error("Sapien is not installed, please install it first")
            raise e
    elif sim == SimType.MUJOCO:
        try:
            from metasim.sim.mujoco import MujocoEnv

            return MujocoEnv
        except ImportError as e:
            log.error("Mujoco is not installed, please install it first")
            raise e
    elif sim == SimType.BLENDER:
        try:
            from metasim.sim.blender import BlenderEnv

            return BlenderEnv
        except ImportError as e:
            log.error("Blender is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN3:
        try:
            from metasim.sim.sapien import Sapien3Env

            return Sapien3Env
        except ImportError as e:
            log.error("Sapien3 is not installed, please install it first")
            raise e
    elif sim == SimType.MJX:
        try:
            from metasim.sim.mjx import MJXEnv

            return MJXEnv
        except ImportError as e:
            log.error("mjx is not installed, please install it first")
            raise e
    else:
        raise ValueError(f"Invalid simulator type: {sim}")


def get_task(task_id: str) -> BaseTaskCfg:
    """Get the task cfg instance from the task id.

    Args:
        task_id: The id of the task.

    Returns:
        The task cfg instance.
    """
    if ":" in task_id:
        prefix, task_name = task_id.split(":")
    else:
        prefix, task_name = None, task_id

    if is_camel_case(task_name):
        task_name_camel = task_name
        task_name_snake = to_snake_case(task_name)
    elif is_snake_case(task_name):
        task_name_camel = to_camel_case(task_name)
        task_name_snake = task_name
    else:
        raise ValueError(f"Invalid task name: {task_id}, should be in either camel case or snake case")

    import_path = f"metasim.cfg.tasks.{prefix}" if prefix is not None else "metasim.cfg.tasks"
    module = importlib.import_module(import_path)
    task_cls = getattr(module, f"{task_name_camel}Cfg")
    gym.register(
        task_name_camel,
        entry_point="metasim.scripts.train_ppo_vec:MetaSimVecEnv",
        kwargs={"task_name": task_name},
    )
    gym.register(
        task_name_snake,
        entry_point="metasim.scripts.train_ppo_vec:MetaSimVecEnv",
        kwargs={"task_name": task_name},
    )
    return task_cls()


def get_robot(robot_name: str) -> BaseRobotCfg:
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
    module = importlib.import_module("metasim.cfg.robots")
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
        module = importlib.import_module("metasim.cfg.scenes")
        scene_cls = getattr(module, f"{scene_name}Cfg")
        return scene_cls()
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Scene {scene_name} not found: {e}") from e
