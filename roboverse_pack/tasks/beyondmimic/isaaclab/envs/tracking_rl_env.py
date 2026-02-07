from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from metasim.integrations.isaaclab.compat.registry import register_manager_based_cfg_task
from metasim.scenario.scenario import ScenarioCfg

if TYPE_CHECKING:
    from roboverse_learn.rl.configs.rsl_rl.ppo_tracking import RslRlPPOTrackingConfig


def _cfg_factory(args: RslRlPPOTrackingConfig, scenario: ScenarioCfg, device: Any):
    """Build the IsaacLab manager-based env cfg (pure factory; no simulator side-effects)."""
    import roboverse_pack.tasks.beyondmimic.isaaclab.configs.flat_env_cfg as flat_env_cfg

    cfg = flat_env_cfg.G1FlatEnvCfg()
    cfg.scene.num_envs = scenario.num_envs
    cfg.seed = args.train_cfg["seed"]
    cfg.sim.device = args.device if getattr(args, "device", None) else cfg.sim.device
    cfg.commands.motion.motion_file = args.motion_file
    return cfg


# Register as a MetaSim handler-backed manager-based task (no IsaacLab SimulationContext).
from roboverse_pack.tasks.beyondmimic.metasim.envs.tracking_g1 import TrackingG1Task

_SCENARIO_TEMPLATE = copy.deepcopy(TrackingG1Task.scenario)

TrackingRLEnv = register_manager_based_cfg_task(
    "motion-tracking-isaaclab",
    _cfg_factory,
    scenario_template=_SCENARIO_TEMPLATE,
)
