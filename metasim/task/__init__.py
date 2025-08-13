"""Auto-import all submodules in the tasks package.

This ensures that all tasks decorated with @register_task are registered
when the package is imported.
"""

from __future__ import annotations

# Auto-discover and import all submodules in this package so that
# tasks decorated with @register_task are registered on import.
from importlib import import_module
from pathlib import Path


def _auto_import_submodules() -> None:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = __name__
    for py_file in pkg_dir.glob("*.py"):
        module_name = py_file.stem
        if module_name in {"__init__"}:
            continue
        # Import submodule (e.g., roboverse_learn.tasks.humanoid_wrapper)
        try:
            import_module(f"{pkg_name}.{module_name}")
        except Exception as exc:  # pragma: no cover
            pass


_auto_import_submodules()
