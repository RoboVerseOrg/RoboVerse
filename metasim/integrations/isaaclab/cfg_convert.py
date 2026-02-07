from __future__ import annotations

from typing import Any

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg


def scenario_from_isaaclab_cfg(
    env_cfg: Any,
    *,
    simulator: str = "isaacsim",
    headless: bool | None = None,
) -> ScenarioCfg:
    """Best-effort conversion from an IsaacLab env cfg to a MetaSim ScenarioCfg.

    Phase 1 scope:
    - Convert runtime knobs (num_envs/env_spacing/decimation/dt/headless).
    - Do NOT attempt to fully convert assets (robots/objects/sensors) yet.

    Args:
        env_cfg: IsaacLab environment config (e.g. `ManagerBasedRLEnvCfg`).
        simulator: Scenario simulator string. Defaults to "isaacsim".
        headless: Optional override. If None, keeps ScenarioCfg default.
    """
    scenario = ScenarioCfg()
    scenario.simulator = simulator

    if headless is not None:
        scenario.headless = headless

    # Scene knobs
    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is not None:
        if hasattr(scene_cfg, "num_envs"):
            scenario.num_envs = int(scene_cfg.num_envs)
        if hasattr(scene_cfg, "env_spacing"):
            scenario.env_spacing = float(scene_cfg.env_spacing)

    # Control knobs
    if hasattr(env_cfg, "decimation"):
        scenario.decimation = int(env_cfg.decimation)

    # Simulation dt
    sim_cfg = getattr(env_cfg, "sim", None)
    dt = getattr(sim_cfg, "dt", None) if sim_cfg is not None else None
    if dt is not None:
        scenario.sim_params = SimParamCfg(dt=float(dt))

    return scenario
