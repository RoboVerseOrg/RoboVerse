"""Task registry with lazy import.

Rules
-----
1.  Task name *must* map directly to a module path:
        "<base_pkg>.<task_name>"
    e.g.  "humanoid.walk"  ->  "roboverse_pack.tasks.humanoid.walk"
2.  Name lookup is case-insensitive. Duplicate names are disallowed.
3.  After the first import the class stays cached in ``TASK_REGISTRY``.

The file is ruff-compliant (E, F, D).
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

from loguru import logger as log

from metasim.task.base import BaseTaskEnv

TASK_REGISTRY: dict[str, type[BaseTaskEnv]] = {}

_BASE_PKGS: Final[list[str]] = [
    "metasim.example.example_pack.tasks",
    "roboverse_pack.tasks",
]


def register_task(*names: str):
    """Decorator: register a ``BaseTaskEnv`` subclass under one or more names."""
    if not names:
        raise ValueError("At least one name is required.")

    def _wrap(cls: type[BaseTaskEnv]):
        if not issubclass(cls, BaseTaskEnv):
            raise TypeError(f"{cls.__name__} is not a BaseTaskEnv")

        for n in names:
            key = n.strip().lower()
            if not key:
                raise ValueError("Task name cannot be empty.")
            if (prev := TASK_REGISTRY.get(key)) and prev is not cls:
                raise ValueError(f"Name '{key}' already used by {prev.__name__}")
            TASK_REGISTRY[key] = cls
        return cls

    return _wrap


def _try_import(name: str) -> None:
    """Import ``<base_pkg>.<name>`` once.

    Task discovery relies on the *name = module suffix* convention.
    If your task is ``humanoid.walk``, its module **must** live at
    ``roboverse_pack.tasks.humanoid.walk`` (or the first matching base
    package listed in ``_BASE_PKGS``).

    Parameters
    ----------
    name :
        Task name, case-insensitive. Dots are preserved.

    Notes:
    -----
    * Stops at the first successful import.
    * Logs a **warning** when a module is absent, and an **error** for any
      other import failure.
    """
    for base in _BASE_PKGS:
        mod_path = f"{base}.{name}"
        try:
            import_module(mod_path)
            return  # success
        except ModuleNotFoundError as exc:
            log.warning(f"Module '{mod_path}' not found: {exc}")
            continue
        except Exception as exc:
            log.error(f"Import error for '{mod_path}': {exc}")


def get_task_class(name: str) -> type[BaseTaskEnv]:
    """Return the task class; import lazily if needed."""
    key = name.strip().lower()

    if (cls := TASK_REGISTRY.get(key)) is not None:
        return cls

    _try_import(key)

    if (cls := TASK_REGISTRY.get(key)) is not None:
        return cls

    available = ", ".join(sorted(TASK_REGISTRY)) or "<none>"
    raise KeyError(f"Unknown task '{name}'. Known: {available}")


def list_tasks() -> list[str]:
    """Return all registered task names (sorted)."""
    return sorted(TASK_REGISTRY)
