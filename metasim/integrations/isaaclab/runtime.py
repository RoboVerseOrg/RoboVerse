from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger as log


@dataclass(frozen=True)
class IsaacLabRuntimeContext:
    """Process-global Isaac Sim / IsaacLab app context.

    This exists to prevent multiple IsaacLab `AppLauncher` instances from being created
    when mixing MetaSim and IsaacLab-style tasks in the same Python process.
    """

    simulation_app: Any
    headless: bool
    enable_cameras: bool


_LOCK = threading.RLock()
_CONTEXT: IsaacLabRuntimeContext | None = None
_REFCOUNT: int = 0


class _SimulationAppProxy:
    """Proxy to make `simulation_app.close()` ref-counted.

    Some IsaacLab tasks call `self.simulation_app.close()` unconditionally in `close()`.
    When the app is shared across tasks, we must avoid closing it while other tasks still
    depend on it. This proxy turns `.close()` into a reference release.
    """

    def __init__(self, app: Any):
        self._app = app

    def close(self) -> None:
        """Release one runtime reference; close the real app on final release."""
        release_isaaclab_app()

    def __getattr__(self, name: str):
        return getattr(self._app, name)


def ensure_isaaclab_source_tree() -> bool:
    """Ensure the full IsaacLab python source tree is importable.

    Some Isaac Sim distributions ship a lightweight `isaaclab` bootstrap package whose
    top-level import works, but most submodules (e.g., `isaaclab.utils`, `isaaclab.envs`)
    live under `isaaclab/source/isaaclab/isaaclab` and are not on `isaaclab.__path__`.

    This helper appends that directory to `isaaclab.__path__` when present.

    Returns:
        True if the path was added, False if no change was needed/possible.
    """
    try:
        import isaaclab
    except Exception:  # pragma: no cover (requires IsaacLab runtime)
        return False

    try:
        base = Path(getattr(isaaclab, "__file__", "")).resolve().parent
    except Exception:  # pragma: no cover - extremely defensive
        return False

    candidate = base / "source" / "isaaclab" / "isaaclab"
    if not candidate.exists():
        return False

    pkg_path = getattr(isaaclab, "__path__", None)
    if pkg_path is None:
        return False

    candidate_str = str(candidate)
    if candidate_str in list(pkg_path):
        return False

    try:
        pkg_path.append(candidate_str)
    except Exception:  # pragma: no cover - path object is odd/unexpected
        return False

    log.info("IsaacLab bootstrap detected; added source tree to isaaclab.__path__: {}", candidate_str)
    return True


def ensure_isaaclab_app(*, headless: bool, enable_cameras: bool) -> IsaacLabRuntimeContext:
    """Ensure an IsaacLab simulation app exists and return the shared context.

    Notes:
    - The first call decides `headless`/`enable_cameras` for the process.
    - Subsequent calls validate compatibility and bump a reference counter.
    """
    global _CONTEXT, _REFCOUNT

    with _LOCK:
        if _CONTEXT is None:
            try:
                from isaaclab.app import AppLauncher
            except ImportError as exc:  # pragma: no cover (requires IsaacLab runtime)
                raise ImportError("IsaacLab is required to launch the Isaac Sim app.") from exc

            parser = argparse.ArgumentParser()
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless = headless
            args.enable_cameras = enable_cameras

            app_launcher = AppLauncher(args)
            _CONTEXT = IsaacLabRuntimeContext(
                simulation_app=app_launcher.app,
                headless=headless,
                enable_cameras=enable_cameras,
            )
            _REFCOUNT = 0
            log.info(
                "IsaacLab runtime created (headless={}, enable_cameras={}).",
                headless,
                enable_cameras,
            )
        else:
            # `headless` mismatch is usually harmless but indicates inconsistent usage.
            if headless != _CONTEXT.headless:
                log.warning(
                    "IsaacLab runtime already exists (headless={}); requested headless={}. Using existing runtime.",
                    _CONTEXT.headless,
                    headless,
                )
            # Cameras cannot reliably be enabled after launch. Enforce strictness.
            if enable_cameras and not _CONTEXT.enable_cameras:
                raise RuntimeError(
                    "IsaacLab runtime was launched with enable_cameras=False but a caller requested enable_cameras=True. "
                    "Restart the process with cameras enabled, or avoid requesting cameras."
                )

        _REFCOUNT += 1
        return IsaacLabRuntimeContext(
            simulation_app=_SimulationAppProxy(_CONTEXT.simulation_app),
            headless=_CONTEXT.headless,
            enable_cameras=_CONTEXT.enable_cameras,
        )


def release_isaaclab_app(*, force: bool = False) -> None:
    """Release one reference to the shared app; close on final release.

    Args:
        force: If True, closes immediately and clears the context.
    """
    global _CONTEXT, _REFCOUNT

    with _LOCK:
        if _CONTEXT is None:
            return

        if force:
            _REFCOUNT = 1

        _REFCOUNT = max(0, _REFCOUNT - 1)
        if _REFCOUNT > 0:
            return

        try:
            log.info("Closing IsaacLab runtime (final release).")
            _CONTEXT.simulation_app.close()
        finally:
            _CONTEXT = None
            _REFCOUNT = 0
