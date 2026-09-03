from __future__ import annotations

import os
import pkgutil
import sys
from importlib import import_module

from loguru import logger as log

from metasim.task._static_index import StaticIndex, build_static_index
from metasim.task.base import BaseTaskEnv
from metasim.utils.package_discovery import get_package_candidates

# Global registry mapping lowercase names to task wrapper classes
TASK_REGISTRY = {}

# Failures encountered during task discovery: module_name -> short error string.
# Surfaced in get_task_class() KeyError so users see import problems right when
# they look up a missing task, rather than having to enable DEBUG logging.
_DISCOVERY_FAILURES: dict[str, str] = {}

# Static (AST) index of ``@register_task`` names -> module, built once per process; see
# ``metasim/task/_static_index.py``. ``METASIM_TASK_DISCOVERY=eager`` restores the old
# import-everything behaviour (useful when debugging import-time side effects).
_STATIC_INDEX: StaticIndex | None = None
_STATIC_INDEX_KEY: tuple[str, ...] | None = None
"""Package list the cached index was built for; a different configuration (env vars, cwd) rebuilds it."""
_EAGER_DONE = False


def register_task(*names):
    """Class decorator to register a task under one or more names.

    Usage:
        @register_task("humanoid.walk", "walk")
        class WalkTask(...):
            ...
    """
    if not names:
        raise ValueError("At least one name must be provided to register_task().")

    def _decorator(cls):
        if not issubclass(cls, BaseTaskEnv):
            log.warning(f"Register class {cls!r} is not a subclass of BaseTaskEnv")
        for raw_name in names:
            key = raw_name.strip().lower()
            if not key:
                raise ValueError("Task name cannot be empty or whitespace only.")
            existing = TASK_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(f"Task name '{key}' is already registered to {existing.__name__}.")
            TASK_REGISTRY[key] = cls
        return cls

    return _decorator


def _task_packages() -> list[str]:
    """Configured task packages (plus ``_*task.py`` modules in the CWD), in discovery order."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    local_task_modules = [
        os.path.splitext(fname)[0] for fname in os.listdir(cwd) if fname.endswith("task.py") and fname.startswith("_")
    ]
    return get_package_candidates(
        "tasks",
        defaults=["metasim.example.example_pack.tasks"],
        local_modules=local_task_modules,
        cwd=cwd,
    )


def _eager_discovery_requested() -> bool:
    return os.environ.get("METASIM_TASK_DISCOVERY", "").strip().lower() == "eager"


def _static_index() -> StaticIndex:
    """Build (once) the AST index of task names -> module for the configured packages."""
    global _STATIC_INDEX, _STATIC_INDEX_KEY
    packages = _task_packages()
    key = tuple(packages)
    if _STATIC_INDEX is None or _STATIC_INDEX_KEY != key:
        local = [m for m in packages if "." not in m]  # cwd ``_*task.py`` modules come through as bare names
        _STATIC_INDEX = build_static_index(packages, local_modules=local)
        _STATIC_INDEX_KEY = key
        for mod, err in _STATIC_INDEX.parse_failures.items():
            _DISCOVERY_FAILURES.setdefault(mod, err)
            log.warning(f"Task discovery: could not scan '{mod}': {err}")
    return _STATIC_INDEX


def _import_registering_module(module_name: str) -> None:
    """Import one task module so its ``@register_task`` decorators run; record failures."""
    try:
        import_module(module_name)
    except Exception as e:
        err_str = f"{type(e).__name__}: {e}"
        _DISCOVERY_FAILURES[module_name] = err_str
        log.warning(f"Task discovery: skip module '{module_name}': {err_str}")


def _discover_task_modules() -> None:
    """Import configured task packages so @register_task decorators run (eager fallback)."""
    global _EAGER_DONE
    if _EAGER_DONE and TASK_REGISTRY:  # an emptied registry (tests, reloads) re-runs discovery
        return
    _EAGER_DONE = True
    packages_to_scan = _task_packages()

    for pkg_name in packages_to_scan:
        try:
            # Import the root package
            pkg = import_module(pkg_name)
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}"
            _DISCOVERY_FAILURES[pkg_name] = err_str
            log.error(f"Task discovery: failed to import package '{pkg_name}': {err_str}")
            continue

        try:
            pkg_path = getattr(pkg, "__path__", None)
            if pkg_path is None:
                continue

            # Scan and import all submodules
            for _finder, module_name, _is_pkg in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
                try:
                    import_module(module_name)
                except Exception as e:
                    err_str = f"{type(e).__name__}: {e}"
                    _DISCOVERY_FAILURES[module_name] = err_str
                    # Log at WARNING (not DEBUG): a discovery failure makes
                    # tasks disappear silently from the registry — the user
                    # then can't run the affected task with no clue why. The
                    # message is also surfaced inside any KeyError raised by
                    # ``get_task_class`` so users hitting an unknown-task
                    # error see related discovery failures.
                    log.warning(f"Task discovery: skip module '{module_name}': {err_str}")
        except Exception as e:
            log.error(f"Task discovery: error scanning package '{pkg_name}': {e}")


def get_task_class(name: str) -> type[BaseTaskEnv]:
    """Return the task wrapper class registered under the given name.

    Name lookup is case-insensitive. Only the module that registers ``name`` is imported (found
    through the static index); the eager import of every task module happens only when the name is
    unknown to the index or ``METASIM_TASK_DISCOVERY=eager`` is set.
    """
    key = name.strip().lower()
    if key in TASK_REGISTRY:
        return TASK_REGISTRY[key]
    if _eager_discovery_requested():
        _discover_task_modules()
    else:
        index = _static_index()
        module_name = index.names.get(key)
        if module_name is not None:
            _import_registering_module(module_name)
        if key not in TASK_REGISTRY:
            # dynamic registrations or a name the index could not see: last resort, import everything
            _discover_task_modules()
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY.keys())) or "<none>"
        msg = f"Unknown task '{name}'. Registered tasks: {available}"
        if _DISCOVERY_FAILURES:
            # Surface import errors so users can see if the task they want
            # failed to register because its module raised during import.
            failures = "\n  ".join(f"{m}: {e}" for m, e in sorted(_DISCOVERY_FAILURES.items()))
            msg += f"\n\n{len(_DISCOVERY_FAILURES)} task module(s) failed to import during discovery:\n  {failures}"
        raise KeyError(msg) from exc


def list_tasks():
    """List all task names (sorted) without importing the task modules.

    Names come from the static index plus anything registered so far; modules whose
    ``register_task`` arguments are not string literals are imported to learn their names.
    """
    if _eager_discovery_requested():
        _discover_task_modules()
        return sorted(TASK_REGISTRY.keys())
    index = _static_index()
    for module_name in sorted(index.dynamic_modules):
        _import_registering_module(module_name)
    return sorted(set(index.names) | set(TASK_REGISTRY.keys()))
