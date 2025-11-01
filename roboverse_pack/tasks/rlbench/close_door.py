from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_door", "close_door", "franka.close_door")
class CloseDoorTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="door_frame",
                usd_path="roboverse_data/assets/rlbench/close_door/door_frame/usd/door_frame.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_door/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.open_door", "open_door", "franka.open_door")
class OpenDoorTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="door_frame",
                usd_path="roboverse_data/assets/rlbench/close_door/door_frame/usd/door_frame.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_door/v2/franka_v2.pkl.gz"
    # TODO: add checker
