from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_wine_bottle", "open_wine_bottle", "franka.open_wine_bottle")
class OpenWineBottleTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="bottle",
                usd_path="roboverse_data/assets/rlbench/open_wine_bottle/bottle/usd/bottle.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_wine_bottle/v2/franka_v2.pkl.gz"
    # TODO: add checker
