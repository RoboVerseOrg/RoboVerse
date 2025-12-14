"""Humanoid task package for RoboVerse.

This package exposes environments and task wrappers used for legged robots
and humanoids within the RoboVerse ecosystem.
"""

from __future__ import annotations

import importlib
import traceback
from pathlib import Path


def _import_task_modules() -> None:
    """Eagerly import task modules so @register_task runs, avoiding a full package crawl.

    We only import modules under `locomotion/` to register tasks while steering
    clear of a recursive import of every helper/callback module, which was
    triggering circular-import timing issues.
    """
    pkg_dir = Path(__file__).resolve().parent
    locomotion_dir = pkg_dir / "locomotion"
    if not locomotion_dir.exists():
        return

    for py_file in sorted(locomotion_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue
        try:
            module_name = ".".join((__name__, "locomotion", py_file.with_suffix("").name))
            importlib.import_module(module_name)
        except Exception:
            traceback.print_exc()


_import_task_modules()
