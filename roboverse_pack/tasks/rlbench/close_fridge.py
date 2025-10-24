from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_fridge", "close_fridge", "franka.close_fridge")
class CloseFridgeTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="fridge_base",
                usd_path="roboverse_data/assets/rlbench/close_fridge/fridge_base/usd/fridge_base.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_fridge/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.open_fridge", "open_fridge", "franka.open_fridge")
class OpenFridgeTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="fridge_base",
                usd_path="roboverse_data/assets/rlbench/close_fridge/fridge_base/usd/fridge_base.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_fridge/v2/franka_v2.pkl.gz"
    # TODO: add checker
