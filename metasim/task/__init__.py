"""Task registry, base env types, and optional Gymnasium integration.

Submodules here are not bulk-imported on package load. Discovery runs only when
you call ``get_task_class`` / ``list_tasks`` or ``register_all_tasks_with_gym``
(see ``metasim.register_gym_envs``), so sim-only scripts stay lightweight.
"""

from __future__ import annotations
