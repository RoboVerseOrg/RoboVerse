"""Task collection: get_started.

Auto-import all submodules in this package so task classes decorated with
@register_task are registered when importing `tasks.get_started`.
"""

from __future__ import annotations

import traceback
from importlib import import_module
from pathlib import Path


def _auto_import_submodules() -> None:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = __name__

    # Recursively import all .py files under this package (excluding __init__.py)
    for py_file in pkg_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            rel = py_file.relative_to(pkg_dir).with_suffix("")
            dotted = ".".join(rel.parts)
            import_module(f"{pkg_name}.{dotted}")
        except Exception as e:
            traceback.print_exc()


_auto_import_submodules()
