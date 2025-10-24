from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.push_buttons", "push_buttons", "franka.push_buttons")
class PushButtonsTask(RLBenchTask):
    max_episode_steps = 500
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="push_buttons_target0",
                usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target0/usd/push_buttons_target0.usd",
            ),
            ArticulationObjCfg(
                name="push_buttons_target1",
                usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target1/usd/push_buttons_target1.usd",
            ),
            ArticulationObjCfg(
                name="push_buttons_target2",
                usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target2/usd/push_buttons_target2.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/push_buttons/v2/franka_v2.pkl.gz"
    # TODO: add checker
