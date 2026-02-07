from __future__ import annotations

from typing import Any, Callable

from loguru import logger as log

from metasim.integrations.isaaclab.runtime import ensure_isaaclab_app, ensure_isaaclab_source_tree
from metasim.integrations.isaaclab.shim import ensure_isaaclab_shim
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .env import HandlerBackedManagerBasedRLEnv


def _disable_debug_vis(cfg: Any, *, simulator: str | None) -> None:
    """Best-effort: disable IsaacSim-only visualizers on non-isaacsim backends."""
    if simulator == "isaacsim":
        return

    # Common IsaacLab patterns:
    # - term cfgs have `debug_vis`
    # - sensor cfgs have `debug_vis`
    visited: set[int] = set()

    def _walk(obj: Any):
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        if hasattr(obj, "debug_vis"):
            try:
                obj.debug_vis = False
            except Exception:
                pass

        if isinstance(obj, (list, tuple)):
            for x in obj:
                _walk(x)
            return
        if isinstance(obj, dict):
            for x in obj.values():
                _walk(x)
            return

        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            # Avoid descending into modules/functions/types
            if callable(val) or isinstance(val, type):
                continue
            if isinstance(val, (int, float, str, bool, bytes)):
                continue
            _walk(val)

    _walk(cfg)


def register_manager_based_cfg_task(
    name: str,
    cfg_factory: Callable[[Any, ScenarioCfg, Any], Any],
    *,
    scenario_template: ScenarioCfg | None = None,
    asset_name_map: dict[str, str] | None = None,
    strict: bool = False,
):
    """Register an IsaacLab manager-based task via a pure cfg factory.

    The returned class is a MetaSim-native task (subclass of BaseTaskEnv), so it
    can be instantiated through `metasim.task.factory.make_task_env(...)`.
    """

    @register_task(name)
    class _CfgTask(HandlerBackedManagerBasedRLEnv):
        scenario = scenario_template or ScenarioCfg()

        def __init__(self, scenario: ScenarioCfg, args: Any, device=None, **kwargs: Any):
            self._isaaclab_runtime = None
            # IsaacLab importability rules:
            # - For non-isaacsim backends, allow importing IsaacLab-style configs even when the real
            #   `isaaclab` package (and USD/pxr) isn't available by installing a pure-Python shim.
            # - For isaacsim backend, do not install the shim; fail fast if IsaacLab isn't importable.
            if scenario.simulator == "isaacsim":
                # IsaacLab is required for the isaacsim handler, and many IsaacLab configs import `isaaclab.*`
                # modules that depend on Omniverse (`omni.*`). Ensure the shared SimulationApp exists before
                # importing task configs.
                ensure_isaaclab_source_tree()
                self._isaaclab_runtime = ensure_isaaclab_app(
                    headless=bool(getattr(scenario, "headless", True)),
                    enable_cameras=bool(getattr(scenario, "cameras", [])),
                )
            else:
                installed = ensure_isaaclab_shim()
                if installed:
                    log.info(
                        "Installed IsaacLab shim for simulator '%s' to import IsaacLab-style configs.",
                        scenario.simulator,
                    )

            cfg = cfg_factory(args, scenario, device)
            _disable_debug_vis(cfg, simulator=scenario.simulator)
            super().__init__(
                scenario=scenario,
                cfg=cfg,
                args=args,
                device=device,
                asset_name_map=asset_name_map,
                strict=strict,
                **kwargs,
            )

    return _CfgTask
