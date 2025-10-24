from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_shoes_in_box", "put_shoes_in_box", "franka.put_shoes_in_box")
class PutShoesInBoxTask(RLBenchTask):
    max_episode_steps = 600
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="box_base",
                usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/box_base/usd/box_base.usd",
            ),
            RigidObjCfg(
                name="shoe1_visual",
                usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/shoe1_visual/usd/shoe1_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="shoe2_visual",
                usd_path="roboverse_data/assets/rlbench/put_shoes_in_box/shoe2_visual/usd/shoe2_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_shoes_in_box/v2/franka_v2.pkl.gz"
    # TODO: add checker
