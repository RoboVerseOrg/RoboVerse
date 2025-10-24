from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_window", "open_window", "franka.open_window")
class OpenWindowTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="window_main",
                usd_path="roboverse_data/assets/rlbench/open_window/window_main/usd/window_main.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_window/v2/franka_v2.pkl.gz"
    # TODO: add checker
