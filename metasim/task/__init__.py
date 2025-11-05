"""Auto-import all submodules in the tasks package.

This ensures that all tasks decorated with @register_task are registered
when the package is imported.
"""

from __future__ import annotations

# After tasks are discoverable, automatically register all tasks with Gymnasium
# so users can call gymnasium.make / gymnasium.make_vec without manual steps.
try:
    from .gym_registration import register_all_tasks_with_gym as _register_all_tasks_with_gym
    from .registry import _discover_task_modules

    _discover_task_modules()
    _register_all_tasks_with_gym()
except Exception:
    # Best-effort: scripts can still call register_task_with_gym on-demand.
    pass
