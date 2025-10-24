from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_grill", "open_grill", "franka.open_grill")
class OpenGrillTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="grill",
                usd_path="roboverse_data/assets/rlbench/open_grill/grill/usd/grill.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_grill/v2/franka_v2.pkl.gz"
    # TODO: add checker
