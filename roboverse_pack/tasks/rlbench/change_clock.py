from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.change_clock", "change_clock", "franka.change_clock")
class ChangeClockTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="clock",
                usd_path="roboverse_data/assets/rlbench/change_clock/clock/usd/clock.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/change_clock/v2/franka_v2.pkl.gz"
    # TODO: add checker
