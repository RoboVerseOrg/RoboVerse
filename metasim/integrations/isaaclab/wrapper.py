from __future__ import annotations

import inspect
from typing import Any

from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg

from .runtime import IsaacLabRuntimeContext, ensure_isaaclab_app


def _supports_runtime_kwarg(task_cls: type) -> bool:
    """Return True if the task's __init__ supports a `runtime=` keyword argument."""
    try:
        sig = inspect.signature(task_cls.__init__)
    except (TypeError, ValueError):
        return False
    return "runtime" in sig.parameters


def make_isaaclab_task_env(
    task_cls: type,
    *,
    scenario: ScenarioCfg,
    args: Any,
    device: Any,
    enable_cameras: bool | None = None,
    **kwargs: Any,
):
    """Instantiate an IsaacLab-style task environment in a MetaSim-friendly way.

    Strategy:
    - If the task supports `runtime=...`, acquire a shared IsaacLab runtime and pass it in.
      This prevents duplicate `AppLauncher` instances and enables ref-counted shutdown.
    - Otherwise, instantiate the task normally and let it manage its own launch.

    Args:
        task_cls: The task class to instantiate (usually from `metasim.task.registry.get_task_class`).
        scenario: MetaSim `ScenarioCfg` used for runtime knobs (num_envs/headless/sim selection).
        args: CLI/config object passed through from training scripts.
        device: Torch device (or string) passed through from training scripts.
        enable_cameras: If None, inferred from `scenario.cameras`.
        kwargs: Extra args forwarded to the task constructor.
    """
    if enable_cameras is None:
        enable_cameras = bool(getattr(scenario, "cameras", []))

    runtime: IsaacLabRuntimeContext | None = None
    if _supports_runtime_kwarg(task_cls):
        runtime = ensure_isaaclab_app(headless=scenario.headless, enable_cameras=enable_cameras)
        return task_cls(scenario=scenario, args=args, device=device, runtime=runtime, **kwargs)

    log.warning(
        "IsaacLab task %s does not accept `runtime=`; instantiating without shared app context. "
        "To make it MetaSim-compatible, add an optional `runtime` kwarg and use it in `_launch()`.",
        getattr(task_cls, "__name__", str(task_cls)),
    )
    return task_cls(scenario=scenario, args=args, device=device, **kwargs)
