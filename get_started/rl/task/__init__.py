from __future__ import annotations

# Expose registry helpers
from roboverse_learn.tasks.registry import TASK_REGISTRY, get_task_class, list_tasks, load_task, register_task

__all__ = [
    "TASK_REGISTRY",
    "get_task_class",
    "list_tasks",
    "load_task",
    "register_task",
]
