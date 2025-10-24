from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.close_laptop_lid", "close_laptop_lid", "franka.close_laptop_lid")
class CloseLaptopLidTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="base",
                usd_path="roboverse_data/assets/rlbench/close_laptop_lid/base/usd/base.usd",
            ),
            RigidObjCfg(
                name="laptop_holder",
                usd_path="roboverse_data/assets/rlbench/close_laptop_lid/laptop_holder/usd/laptop_holder.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/close_laptop_lid/v2/franka_v2.pkl.gz"
    # TODO: add checker
