"""RoboVerse task pack: robosuite integration.

Two entry points are registered when this package is imported:

* **Native MetaSim ports** (``robosuite.lift``, ``robosuite.stack`` …) — task
  classes that load robosuite's *own* compiled MJCF and step it on MetaSim's
  ``MujocoHandler``. Physics is bit-for-bit identical to robosuite (see the
  ``tools/robosuite_integration`` parity harness); robosuite's controller,
  reward, observation and success logic are reused so behaviour — and therefore
  any policy's performance — matches the native env exactly.

* **Passthrough** (``Robosuite/<EnvName>`` in the gym registry) — thin wrappers
  around ``robosuite.make`` for users who want the unmodified native env through
  the same ``gym.make`` surface as the rest of RoboVerse.

robosuite is an optional dependency; if it (or its assets) cannot be imported
the registration is skipped quietly so the rest of the pack still loads.
"""

from __future__ import annotations

import traceback
from importlib import import_module
from pathlib import Path


def _auto_import_submodules() -> None:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = __name__
    for py_file in pkg_dir.rglob("*.py"):
        if py_file.name == "__init__.py" or py_file.name.startswith("_passthrough"):
            continue
        rel = py_file.relative_to(pkg_dir).with_suffix("")
        try:
            import_module(f"{pkg_name}.{'.'.join(rel.parts)}")
        except Exception:  # pragma: no cover - robosuite optional
            traceback.print_exc()


_auto_import_submodules()

try:
    from roboverse_pack.tasks.robosuite._passthrough import register_robosuite_passthrough

    register_robosuite_passthrough()
except Exception:  # pragma: no cover - robosuite optional
    pass
