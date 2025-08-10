from __future__ import annotations

from roboverse_learn.tasks.base import BaseTaskWrapper
from scenario_cfg.scenario import ScenarioCfg

# Global registry mapping lowercase names to task wrapper classes
TASK_REGISTRY = {}


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
        if not issubclass(cls, BaseTaskWrapper):
            raise TypeError(f"Can only register subclasses of BaseTaskWrapper, got: {cls!r}")
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


def get_task_class(name: str):
    """Return the task wrapper class registered under the given name.

    Name lookup is case-insensitive.
    """

    key = name.strip().lower()
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from exc


def list_tasks():
    """List all registered task names (sorted)."""

    return sorted(TASK_REGISTRY.keys())


def load_task(name: str, scenario: ScenarioCfg, *args, **kwargs) -> BaseTaskWrapper:
    """Instantiate a registered task by name.

    Args:
        name: Registered task name (case-insensitive)
        scenario: The `ScenarioCfg` to construct the task with
        *args, **kwargs: Forwarded to the task class constructor
    """

    cls = get_task_class(name)
    return cls(scenario, *args, **kwargs)
