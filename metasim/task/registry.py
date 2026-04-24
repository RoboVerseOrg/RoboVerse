from __future__ import annotations

import os
import pkgutil
import sys
from importlib import import_module

from loguru import logger as log

from metasim.task.base import BaseTaskEnv

# Global registry mapping lowercase names to task wrapper classes
TASK_REGISTRY = {}

# Failures encountered during task discovery: module_name -> short error string.
# Surfaced in get_task_class() KeyError so users see import problems right when
# they look up a missing task, rather than having to enable DEBUG logging.
_DISCOVERY_FAILURES: dict[str, str] = {}


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


def _discover_task_modules() -> None:
    """Import modules from known task packages so @register_task runs.

    Scans these packages (if available):
      - metasim.example.example_pack.tasks
      - roboverse_pack.tasks


    Safe to call multiple times; import errors are ignored to avoid breaking
    discovery due to one bad module.
    """
    packages_to_scan = [
        "metasim.example.example_pack.tasks",
        "roboverse_pack.tasks",
    ]
    if os.environ.get("METASIM_TASK_PACKAGES", None):
        packages = os.environ["METASIM_TASK_PACKAGES"].split(":")
        log.info(f"Scanning additional task packages: {packages}")
        packages_to_scan.extend(packages)

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    for fname in os.listdir(cwd):
        if fname.endswith("task.py") and fname.startswith("_"):
            modname = os.path.splitext(fname)[0]
            packages_to_scan.append(modname)
    for pkg_name in packages_to_scan:
        try:
            # Import the root package
            pkg = import_module(pkg_name)
        except Exception as e:
            log.error(f"Task discovery: failed to import package '{pkg_name}': {e}")
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
                    log.debug(f"Task discovery: skip module '{module_name}': {err_str}")
        except Exception as e:
            log.error(f"Task discovery: error scanning package '{pkg_name}': {e}")


def get_task_class(name: str) -> type[BaseTaskEnv]:
    """Return the task wrapper class registered under the given name.

    Name lookup is case-insensitive.
    """
    # ensure modules are imported so registry is populated
    if not TASK_REGISTRY:
        _discover_task_modules()

    key = name.strip().lower()
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
    """List all registered task names (sorted)."""
    if not TASK_REGISTRY:
        _discover_task_modules()
    return sorted(TASK_REGISTRY.keys())
