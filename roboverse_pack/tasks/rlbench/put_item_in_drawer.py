from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_item_in_drawer", "put_item_in_drawer", "franka.put_item_in_drawer")
class PutItemInDrawerTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="drawer_frame",
                usd_path="roboverse_data/assets/rlbench/put_item_in_drawer/drawer_frame/usd/drawer_frame.usd",
            ),
            PrimitiveCubeCfg(
                name="item",
                size=[0.04, 0.04, 0.04],
                color=[0.85, 0.85, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_item_in_drawer/v2/franka_v2.pkl.gz"
    # TODO: add checker
