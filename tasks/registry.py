from __future__ import annotations

from importlib import import_module
from pathlib import Path

from scenario_cfg.scenario import ScenarioCfg
from tasks.base import BaseTaskEnv

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
        if not issubclass(cls, BaseTaskEnv):
            raise TypeError(f"Can only register subclasses of BaseTaskEnv, got: {cls!r}")
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
    """Recursively import all modules under `tasks` so @register_task runs.

    Safe to call multiple times; import errors are ignored to avoid breaking
    discovery due to one bad module.
    """
    try:
        base_pkg = __package__.split(".")[0]  # "tasks"
        base_dir = Path(__file__).resolve().parent
        for py_file in base_dir.rglob("*.py"):
            if py_file.name in {"__init__.py", Path(__file__).name}:
                continue
            rel = py_file.relative_to(base_dir).with_suffix("")
            dotted = ".".join((base_pkg, *rel.parts))
            try:
                import_module(dotted)
            except Exception:
                pass
    except Exception:
        pass


def get_task_class(name: str):
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
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from exc


def list_tasks():
    """List all registered task names (sorted)."""
    if not TASK_REGISTRY:
        _discover_task_modules()
    return sorted(TASK_REGISTRY.keys())


def load_task(name, scenario: ScenarioCfg, *args, **kwargs) -> BaseTaskEnv:
    """Instantiate a registered task by name.

    Args:
        name: Registered task name (case-insensitive).
        scenario: The `ScenarioCfg` to construct the task with.
        *args: Additional positional arguments forwarded to the task class constructor.
        **kwargs: Additional keyword arguments forwarded to the task class constructor.

    Returns:
        BaseTaskEnv: The instantiated task environment.
    """
    cls = get_task_class(name)
    return cls(scenario, *args, **kwargs)
