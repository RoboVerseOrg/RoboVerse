"""Unitree RL task package for RoboVerse.

This package exposes environments and task wrappers used for legged robots
and humanoids within the RoboVerse ecosystem.
"""

from __future__ import annotations

import importlib
import traceback
from pathlib import Path


def _auto_import_submodules() -> None:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = __name__

    for py_file in pkg_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            rel = py_file.relative_to(pkg_dir).with_suffix("")
            dotted = ".".join((pkg_name, *rel.parts))
            importlib.import_module(dotted)
        except Exception:
            traceback.print_exc()


_auto_import_submodules()
