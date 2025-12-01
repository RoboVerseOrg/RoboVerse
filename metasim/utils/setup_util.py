"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import importlib
import os
import sys

from loguru import logger as log

from metasim.constants import SimType
from metasim.scenario.grounds import GroundCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scene import SceneCfg
from metasim.sim.parallel import ParallelSimWrapper
from metasim.utils import is_camel_case, is_snake_case, to_camel_case


def get_handler(scenario, device=None):
    """Get a launched simulator handler from scenario configuration.

    This function combines three steps into one:
    1. Get handler class from scenario.simulator
    2. Create handler instance with scenario
    3. Launch the handler

    Args:
        scenario: ScenarioCfg instance containing simulation configuration
        device: current device

    Returns:
        Launched simulator handler ready for use
    """
    # Step 1: Get handler class
    handler_class = get_sim_handler_class(SimType(scenario.simulator))

    # Step 2: Create handler instance
    handler = handler_class(scenario, device)

    # Step 3: Launch handler
    handler.launch()

    return handler


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
    elif sim == SimType.ISAACSIM:
        try:
            from metasim.sim.isaacsim import IsaacsimHandler

            return IsaacsimHandler
        except ImportError as e:
            log.error("IsaacSim is not installed, please install it first")
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

            return ParallelSimWrapper(PybulletHandler)
        except ImportError as e:
            log.error("PyBullet is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN2:
        try:
            from metasim.sim.sapien.sapien2 import Sapien2Handler

            return ParallelSimWrapper(Sapien2Handler)
        except ImportError as e:
            log.error("Sapien is not installed, please install it first")
            raise e
    elif sim == SimType.SAPIEN3:
        try:
            from metasim.sim.sapien.sapien3 import Sapien3Handler

            return ParallelSimWrapper(Sapien3Handler)
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

    # Search across both official and example robot config packages (union)
    candidate_packages = [
        "roboverse_pack.robots",
        "metasim.example.example_pack.robots",
    ]
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    attr_name = f"{RobotName}Cfg"
    errors: list[str] = []

    for fname in os.listdir(cwd):
        if fname.endswith(".py") and not fname.startswith("_"):
            modname = os.path.splitext(fname)[0]
            candidate_packages.append(modname)

    for pkg_name in candidate_packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            errors.append(f"{pkg_name}: import failed ({e})")
            continue

        # Fast path: attribute re-exported from package __init__
        try:
            robot_cls = getattr(pkg, attr_name)
            return robot_cls()
        except AttributeError:
            pass

        except Exception as e:
            errors.append(f"{pkg_name}: scan failed ({e})")

    # Not found in any package
    searched_in = ", ".join(candidate_packages)
    raise ValueError(f"Robot config class '{attr_name}' not found in [{searched_in}]. Errors: {errors}")


def get_scene(scene_name: str) -> SceneCfg:
    """Get a scene configuration by name.

    Args:
        scene_name (str): The name of the scene in either snake_case or CamelCase format.

    Returns:
        SceneCfg: The scene configuration object corresponding to the given scene name.

    Raises:
        ValueError: If the scene name is not found or has invalid format.
    """
    if is_snake_case(scene_name):
        SceneName = to_camel_case(scene_name)
    elif is_camel_case(scene_name):
        SceneName = scene_name
    else:
        raise ValueError(f"Invalid scene name: {scene_name}")

    candidate_packages = [
        "roboverse_pack.scenes",
    ]

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    for fname in os.listdir(cwd):
        if fname.endswith(".py") and not fname.startswith("_"):
            candidate_packages.append(os.path.splitext(fname)[0])

    attr_name = f"{SceneName}Cfg"
    errors: list[str] = []
    for pkg_name in candidate_packages:
        try:
            pkg = importlib.import_module(pkg_name)
            scene_cls = getattr(pkg, attr_name)
            return scene_cls()
        except AttributeError:
            continue
        except Exception as e:
            errors.append(f"{pkg_name}: {e}")
            continue

    raise ValueError(f"Scene config class '{attr_name}' not found in {candidate_packages}. Errors: {errors}")


def get_ground(ground_name: str) -> GroundCfg:
    """Resolve a ground configuration by name."""
    if is_snake_case(ground_name):
        GroundName = to_camel_case(ground_name)
    elif is_camel_case(ground_name):
        GroundName = ground_name
    else:
        raise ValueError(f"Invalid ground name: {ground_name}")

    candidate_packages = [
        "roboverse_pack.grounds",
    ]

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    for fname in os.listdir(cwd):
        if fname.endswith(".py") and not fname.startswith("_"):
            candidate_packages.append(os.path.splitext(fname)[0])

    attr_name = f"{GroundName}Cfg"
    errors: list[str] = []

    for pkg_name in candidate_packages:
        try:
            pkg = importlib.import_module(pkg_name)
            ground_cls = getattr(pkg, attr_name)
            return ground_cls()
        except AttributeError:
            continue
        except Exception as e:
            errors.append(f"{pkg_name}: {e}")

    raise ValueError(f"Ground config class '{attr_name}' not found in {candidate_packages}. Errors: {errors}")
