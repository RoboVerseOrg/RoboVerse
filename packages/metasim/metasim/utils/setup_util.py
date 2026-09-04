"""Sub-module containing utilities for setting up the environment."""

from __future__ import annotations

import ast
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


_LOCAL_BINDINGS: dict[str, tuple[tuple[float, int], frozenset[str], bool, str | None]] = {}
"""path -> (mtime + size, names bound anywhere in the module, can bind any name, parse failure)."""


def _bound_names(tree: ast.AST) -> tuple[set[str], bool]:
    """Names a module's source binds at any depth, and whether it could bind anything.

    Classes, functions, assignment / loop / with / walrus / except targets and imports count (an import
    under ``if TYPE_CHECKING:`` does not: it binds nothing at runtime). A star import, a write through
    ``globals()`` / ``vars()``, ``exec``, ``setattr(sys.modules[...], ...)`` or a module-level
    ``__getattr__`` (PEP 562) can bind any name, so the module is a candidate for every lookup
    (``wildcard``).
    """
    names: set[str] = set()
    wildcard = False
    typing_only = {
        id(node)
        for block in ast.walk(tree)
        if isinstance(block, ast.If) and _is_type_checking_test(block.test)
        for stmt in block.body  # the ``else`` branch runs
        for node in ast.walk(stmt)
    }
    nested = {
        id(node)
        for scope in ast.walk(tree)
        if isinstance(scope, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(scope)
        if node is not scope
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__" and id(node) not in nested:
            wildcard = True  # a module ``__getattr__`` (PEP 562), wherever the module body defines it
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    wildcard = True
                elif id(node) not in typing_only:
                    names.add(alias.asname or alias.name.split(".")[0])
        elif _writes_globals(node) or _binds_dynamically(node):
            wildcard = True
    return names, wildcard


def _is_type_checking_test(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _is_globals_call(node: ast.AST) -> bool:
    """``globals()`` or ``vars()`` (the module namespace when called without arguments)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "globals" or (node.func.id == "vars" and not node.args))
    )


def _binds_dynamically(node: ast.AST) -> bool:
    """``exec(...)`` or ``setattr(sys.modules[...], ...)``: a binding of any name."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return False
    if node.func.id == "exec":
        return True
    if node.func.id == "setattr" and node.args:
        target = node.args[0]
        return (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "modules"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "sys"
        )
    return False


def _writes_globals(node: ast.AST) -> bool:
    """``globals()[...] = ...``, ``globals().update(...)`` / ``.setdefault(...)``: a binding of any name."""
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
        return _is_globals_call(node.value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"update", "setdefault"}
    ):
        return _is_globals_call(node.func.value)
    return False


def _local_modules_defining(cwd: str, attr_name: str) -> tuple[list[str], list[str]]:
    """The top-level ``.py`` modules in ``cwd`` whose source may bind ``attr_name``, and the parse failures.

    Returns ``(modules, failures)``; a failure reads ``"module: not parsed (...)"``. The source is
    parsed, not imported, so a config lookup never executes the other scripts that happen to sit in
    the working directory (``train.py`` used to run on ``get_robot``). A binding at any depth counts
    (``try: from pack import FrankaCfg``, ``def``, loop / with / walrus targets, imports other than under
    ``if TYPE_CHECKING:``), and a module that can bind any name (``import *``, a ``globals()`` write, a
    module-level ``__getattr__``) is always a candidate. Parse
    results are cached per file for the process (keyed by mtime + size); the bytes are handed to
    ``ast`` so a ``coding:`` cookie is honoured as the import system would.
    """
    found: list[str] = []
    failures: list[str] = []
    for fname in sorted(os.listdir(cwd)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(cwd, fname)
        module = fname[:-3]
        try:
            st = os.stat(path)
            stamp = (st.st_mtime, st.st_size)
        except OSError as exc:
            failures.append(f"{module}: not parsed ({type(exc).__name__}: {exc})")
            continue
        entry = _LOCAL_BINDINGS.get(path)
        if entry is None or entry[0] != stamp:
            names: frozenset[str] = frozenset()
            wildcard, failure = False, None
            try:
                with open(path, "rb") as f:
                    bound, wildcard = _bound_names(ast.parse(f.read(), filename=path))
                names = frozenset(bound)
            except (OSError, SyntaxError, ValueError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            entry = (stamp, names, wildcard, failure)
            _LOCAL_BINDINGS[path] = entry
        _, names, wildcard, failure = entry
        if failure is not None:
            failures.append(f"{module}: not parsed ({failure})")
        elif wildcard or attr_name in names:
            found.append(module)
    return found, failures


_SHADOW_WARNED: set[tuple[str, str]] = set()


def _lookup_cfg(attr_name: str, candidate_packages: list[str], cfg_kind: str, *, extra_errors: list[str] = ()):
    """Instantiate ``attr_name`` from the first candidate package that defines it.

    Every candidate is inspected so a name defined in more than one package is reported once (the
    first, highest-precedence package wins; see ``get_package_candidates`` for the order). A silent
    first-match used to hide that MetaSim's example pack was shadowing a content pack's config.
    """
    errors: list[str] = list(extra_errors)  # e.g. a local file that could not be parsed
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
    local_modules, local_errors = _local_modules_defining(cwd, attr_name)
    candidate_packages = get_package_candidates(
        "robots",
        defaults=["metasim.example.example_pack.robots"],
        local_modules=local_modules,
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Robot", extra_errors=local_errors)


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
    local_modules, local_errors = _local_modules_defining(cwd, attr_name)
    candidate_packages = get_package_candidates(
        "scenes",
        local_modules=local_modules,
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Scene", extra_errors=local_errors)


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
    local_modules, local_errors = _local_modules_defining(cwd, attr_name)
    candidate_packages = get_package_candidates(
        "grounds",
        local_modules=local_modules,
        cwd=cwd,
    )
    return _lookup_cfg(attr_name, candidate_packages, "Ground", extra_errors=local_errors)
