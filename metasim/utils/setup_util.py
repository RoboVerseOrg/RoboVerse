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
from metasim.utils.package_discovery import get_package_candidates


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
        # Backward-compat shim: the standalone IsaacLab handler was removed in
        # commit b752bb2 (C3). Existing callers that pass SimType.ISAACLAB
        # are routed to the IsaacSim handler, which uses the ``isaaclab``
        # Python package internally and is the modern successor.
        import warnings

        warnings.warn(
            "SimType.ISAACLAB is deprecated; use SimType.ISAACSIM. "
            "Dispatching to the IsaacSim handler.",
            DeprecationWarning,
            stacklevel=2,
        )
        sim = SimType.ISAACSIM
    if sim == SimType.ISAACGYM:
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
    elif sim == SimType.NEWTON:
        try:
            from metasim.sim.newton import NewtonHandler

            return NewtonHandler
        except ImportError as e:
            log.error("Newton is not installed. Activate newton conda environment: conda activate newton")
            raise e
    else:
        raise ValueError(f"Invalid simulator type: {sim}")


def _local_python_modules(cwd: str) -> list[str]:
    return [
        os.path.splitext(fname)[0]
        for fname in os.listdir(cwd)
        if fname.endswith(".py") and not fname.startswith("_")
    ]


def _lookup_cfg(attr_name: str, candidate_packages: list[str], cfg_kind: str):
    errors: list[str] = []
    for pkg_name in candidate_packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            errors.append(f"{pkg_name}: import failed ({e})")
            continue

        try:
            cfg_cls = getattr(pkg, attr_name)
            return cfg_cls()
        except AttributeError:
            continue
        except Exception as e:
            errors.append(f"{pkg_name}: lookup failed ({e})")

    searched_in = ", ".join(candidate_packages)
    raise ValueError(f"{cfg_kind} config class '{attr_name}' not found in [{searched_in}]. Errors: {errors}")


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

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    attr_name = f"{RobotName}Cfg"
    candidate_packages = get_package_candidates(
        "robots",
        defaults=["metasim.example.example_pack.robots"],
        local_modules=_local_python_modules(cwd),
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Robot")


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

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    attr_name = f"{SceneName}Cfg"
    candidate_packages = get_package_candidates(
        "scenes",
        local_modules=_local_python_modules(cwd),
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Scene")


def get_ground(ground_name: str) -> GroundCfg:
    """Resolve a ground configuration by name."""
    if is_snake_case(ground_name):
        GroundName = to_camel_case(ground_name)
    elif is_camel_case(ground_name):
        GroundName = ground_name
    else:
        raise ValueError(f"Invalid ground name: {ground_name}")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    attr_name = f"{GroundName}Cfg"
    candidate_packages = get_package_candidates(
        "grounds",
        local_modules=_local_python_modules(cwd),
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Ground")
