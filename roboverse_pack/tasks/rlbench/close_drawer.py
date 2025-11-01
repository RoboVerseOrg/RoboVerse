from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_drawer", "close_drawer", "franka.close_drawer")
class CloseDrawerTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="drawer_frame",
                usd_path="roboverse_data/assets/rlbench/close_drawer/drawer_frame/usd/drawer_frame.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_drawer/v2/franka_v2.pkl.gz"
    # TODO: add checker


@register_task("rlbench.open_drawer", "open_drawer", "franka.open_drawer")
class OpenDrawerTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="drawer_frame",
                usd_path="roboverse_data/assets/rlbench/close_drawer/drawer_frame/usd/drawer_frame.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/open_drawer/v2/franka_v2.pkl.gz"
    # TODO: add checker
