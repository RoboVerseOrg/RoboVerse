"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass

from loguru import logger as log

from metasim.constants import SimType
from metasim.scenario.grounds import GroundCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scene import SceneCfg
from metasim.sim.parallel import ParallelSimWrapper
from metasim.utils import is_camel_case, is_snake_case, to_camel_case
from metasim.utils.package_discovery import get_package_candidates


def get_handler(scenario, optional_queries=None):
    """Get a launched simulator handler from scenario configuration.

    This function combines three steps into one:
    1. Get handler class from scenario.simulator
    2. Create handler instance with scenario
    3. Launch the handler

    Args:
        scenario: ScenarioCfg instance containing simulation configuration
        optional_queries: optional ``{name: BaseQueryType}`` dict bound to the
            handler and exposed via ``handler.get_extra()``. (Each handler
            derives its device internally; the previous ``device`` argument was
            forwarded into this same constructor slot, so it was dead when None
            and crashed ``launch()`` when a device string was passed.)

    Returns:
        Launched simulator handler ready for use
    """
    # Step 1: Get handler class
    handler_class = get_sim_handler_class(SimType(scenario.simulator))

    # Step 2: Create handler instance
    handler = handler_class(scenario, optional_queries)

    # Step 3: Launch handler
    handler.launch()

    return handler


@dataclass(frozen=True)
class BackendSpec:
    """How one ``SimType`` is resolved to a handler class.

    ``module``/``cls`` are imported lazily so an uninstalled backend costs nothing until used.
    ``parallel`` wraps single-environment handlers in :class:`~metasim.sim.parallel.ParallelSimWrapper`
    so ``num_envs > 1`` is served by worker processes. ``install_hint`` is logged when the lazy
    import fails.
    """

    module: str
    cls: str
    parallel: bool = False
    install_hint: str = ""


SIM_BACKENDS: dict[SimType, BackendSpec] = {
    SimType.ISAACGYM: BackendSpec(
        "metasim.sim.isaacgym", "IsaacgymHandler", False, "IsaacGym is not installed, please install it first"
    ),
    SimType.ISAACSIM: BackendSpec(
        "metasim.sim.isaacsim",
        "IsaacsimHandler",
        False,
        "IsaacSim backend needs `isaacsim` (pip, Python 3.11 for 5.0) AND Isaac Lab from source: see packages/metasim/requirements/isaacsim5.txt",
    ),
    SimType.GENESIS: BackendSpec(
        "metasim.sim.genesis", "GenesisHandler", False, "Genesis is not installed, please install it first"
    ),
    SimType.PYREP: BackendSpec(
        "metasim.sim.pyrep", "PyrepHandler", False, "PyRep is not installed, please install it first"
    ),
    SimType.PYBULLET: BackendSpec(
        "metasim.sim.pybullet", "PybulletHandler", True, "PyBullet is not installed, please install it first"
    ),
    SimType.SAPIEN2: BackendSpec(
        "metasim.sim.sapien.sapien2", "Sapien2Handler", True, "Sapien is not installed, please install it first"
    ),
    SimType.SAPIEN3: BackendSpec(
        "metasim.sim.sapien.sapien3", "Sapien3Handler", True, "Sapien is not installed, please install it first"
    ),
    SimType.MUJOCO: BackendSpec(
        "metasim.sim.mujoco", "MujocoHandler", True, "Mujoco is not installed, please install it first"
    ),
    SimType.BLENDER: BackendSpec(
        "metasim.sim.blender", "BlenderHandler", False, "Blender is not installed, please install it first"
    ),
    SimType.MJX: BackendSpec("metasim.sim.mjx", "MJXHandler", False, "MJX is not installed, please install it first"),
    SimType.NEWTON: BackendSpec(
        "metasim.sim.newton",
        "NewtonHandler",
        False,
        "Newton is not installed. Activate newton conda environment: conda activate newton",
    ),
    SimType.SUPERDEX: BackendSpec(
        "metasim.sim.superdex",
        "SuperdexHandler",
        True,
        "SuperDex is not installed (Python >= 3.12 wheels): python -m pip install superdex-physics superdex-robotics pyrender",
    ),
}
"""Single source of truth for backend dispatch. Adding a simulator = one ``SimType`` value + one entry here."""


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
            "SimType.ISAACLAB is deprecated; use SimType.ISAACSIM. Dispatching to the IsaacSim handler.",
            DeprecationWarning,
            stacklevel=2,
        )
        sim = SimType.ISAACSIM
    spec = SIM_BACKENDS.get(sim)
    if spec is None:
        raise ValueError(f"Invalid simulator type: {sim}")
    # Version gate before the import: an unsupported simulator release fails here with the
    # installed/supported versions in the message, not with an AttributeError inside the handler.
    from metasim.sim._versions import enforce_backend_versions

    enforce_backend_versions(sim)
    try:
        handler_cls = getattr(importlib.import_module(spec.module), spec.cls)
    except ImportError as e:
        log.error(spec.install_hint)
        raise e
    return ParallelSimWrapper(handler_cls) if spec.parallel else handler_cls


def _local_python_modules(cwd: str) -> list[str]:
    return [
        os.path.splitext(fname)[0] for fname in os.listdir(cwd) if fname.endswith(".py") and not fname.startswith("_")
    ]


_SHADOW_WARNED: set[tuple[str, str]] = set()


def _lookup_cfg(attr_name: str, candidate_packages: list[str], cfg_kind: str):
    """Instantiate ``attr_name`` from the first candidate package that defines it.

    Every candidate is inspected so a name defined in more than one package is reported once (the
    first, highest-precedence package wins; see ``get_package_candidates`` for the order). A silent
    first-match used to hide that MetaSim's example pack was shadowing a content pack's config.
    """
    errors: list[str] = []
    found: list[tuple[str, type]] = []
    for pkg_name in candidate_packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            errors.append(f"{pkg_name}: import failed ({e})")
            continue
        cfg_cls = getattr(pkg, attr_name, None)
        if cfg_cls is not None:
            found.append((pkg_name, cfg_cls))

    if not found:
        searched_in = ", ".join(candidate_packages)
        raise ValueError(f"{cfg_kind} config class '{attr_name}' not found in [{searched_in}]. Errors: {errors}")
    if len(found) > 1 and (attr_name, found[0][0]) not in _SHADOW_WARNED:
        _SHADOW_WARNED.add((attr_name, found[0][0]))
        others = ", ".join(p for p, _ in found[1:])
        log.warning(
            f"{cfg_kind} config '{attr_name}' is defined in several packages; using {found[0][0]} (also in: {others})."
        )
    pkg_name, cfg_cls = found[0]
    try:
        return cfg_cls()
    except Exception as e:
        raise ValueError(f"{cfg_kind} config class '{attr_name}' from {pkg_name} could not be instantiated: {e}") from e


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
