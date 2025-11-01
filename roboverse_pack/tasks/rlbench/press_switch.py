from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.press_switch", "press_switch", "franka.press_switch")
class PressSwitchTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="switch_main",
                usd_path="roboverse_data/assets/rlbench/press_switch/switch_main/usd/switch_main.usd",
            ),
            RigidObjCfg(
                name="task_wall",
                usd_path="roboverse_data/assets/rlbench/press_switch/task_wall/usd/task_wall.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/press_switch/v2/franka_v2.pkl.gz"
    # TODO: add checker
