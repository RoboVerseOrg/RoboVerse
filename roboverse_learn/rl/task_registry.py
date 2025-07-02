"""Task wrapper registry for dynamic task loading.

This module provides a registry system for task wrappers, allowing
dynamic discovery and instantiation of task-specific wrappers based
on task names.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

from .task_wrapper import BaseTaskWrapper

if TYPE_CHECKING:
    pass


class TaskWrapperRegistry:
    """Registry for task wrapper classes."""

    def __init__(self):
        self._registry: dict[str, type[BaseTaskWrapper]] = {}
        self._loaded_modules = set()

    def register(self, task_name: str, wrapper_class: type[BaseTaskWrapper]):
        """Register a task wrapper class.

        Args:
            task_name: Name of the task (e.g., 'isaacgym_envs:AllegroHand')
            wrapper_class: The wrapper class to register
        """
        if not issubclass(wrapper_class, BaseTaskWrapper):
            raise ValueError(f"{wrapper_class} must be a subclass of BaseTaskWrapper")

        self._registry[task_name] = wrapper_class

    def get(self, task_name: str) -> type[BaseTaskWrapper] | None:
        """Get a registered wrapper class.

        Args:
            task_name: Name of the task

        Returns:
            The wrapper class or None if not found
        """
        # Try to load the module if not already loaded
        if task_name not in self._registry:
            self._try_load_task_module(task_name)

        return self._registry.get(task_name)

    def _try_load_task_module(self, task_name: str):
        """Try to load a task module based on naming convention.

        Args:
            task_name: Task name in format 'benchmark:task'
        """
        if ":" not in task_name:
            return

        benchmark, task = task_name.split(":", 1)

        # Convert task name to module name (e.g., AllegroHand -> allegro_hand)
        module_name = self._task_to_module_name(task)

        # Try to import the module
        module_paths = [
            f"roboverse_learn.rl.tasks.{benchmark}.{module_name}",
            f"roboverse_learn.rl.tasks.{module_name}",
        ]

        for module_path in module_paths:
            if module_path in self._loaded_modules:
                continue

            try:
                module = importlib.import_module(module_path)
                self._loaded_modules.add(module_path)

                # Auto-register any BaseTaskWrapper subclasses found
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseTaskWrapper) and obj is not BaseTaskWrapper:
                        # Use the class's registered_name attribute if available
                        registered_name = getattr(obj, "registered_name", task_name)
                        if registered_name not in self._registry:
                            self._registry[registered_name] = obj

                break
            except ImportError:
                continue

    def _task_to_module_name(self, task_name: str) -> str:
        """Convert task name to module name.

        Examples:
            AllegroHand -> allegro_hand
            AnymalTerrain -> anymal_terrain
        """
        # Handle camelCase and PascalCase
        result = []
        for i, char in enumerate(task_name):
            if i > 0 and char.isupper() and task_name[i - 1].islower():
                result.append("_")
            result.append(char.lower())
        return "".join(result)

    def list_registered(self) -> list:
        """List all registered task names."""
        return sorted(self._registry.keys())


# Global registry instance
_task_wrapper_registry = TaskWrapperRegistry()


def register_task_wrapper(task_name: str):
    """Decorator to register a task wrapper class.

    Usage:
        @register_task_wrapper('isaacgym_envs:AllegroHand')
        class AllegroHandTaskWrapper(BaseTaskWrapper):
            ...
    """

    def decorator(cls: type[BaseTaskWrapper]):
        _task_wrapper_registry.register(task_name, cls)
        cls.registered_name = task_name
        return cls

    return decorator


def get_task_wrapper(task_name: str, env, cfg, sim_type: str) -> BaseTaskWrapper | None:
    """Get an instance of a task wrapper.

    Args:
        task_name: Name of the task
        env: The environment to wrap
        cfg: Task configuration
        sim_type: Simulator type

    Returns:
        Instance of the task wrapper or None if not found
    """
    wrapper_class = _task_wrapper_registry.get(task_name)
    if wrapper_class is None:
        return None

    return wrapper_class(env, cfg, sim_type)


def list_available_wrappers() -> list:
    """List all available task wrappers."""
    return _task_wrapper_registry.list_registered()
