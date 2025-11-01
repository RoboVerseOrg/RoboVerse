from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.open_washing_machine", "open_washing_machine", "franka.open_washing_machine")
class OpenWashingMachineTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="washer",
                usd_path="roboverse_data/assets/rlbench/open_washing_machine/washer/usd/washer.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_washing_machine/v2/franka_v2.pkl.gz"
    # TODO: add checker
