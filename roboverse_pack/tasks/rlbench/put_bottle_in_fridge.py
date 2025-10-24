from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.put_bottle_in_fridge", "put_bottle_in_fridge", "franka.put_bottle_in_fridge")
class PutBottleInFridgeTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bottle_visual",
                usd_path="roboverse_data/assets/rlbench/put_bottle_in_fridge/bottle_visual/usd/bottle_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            ArticulationObjCfg(
                name="fridge_base",
                usd_path="roboverse_data/assets/rlbench/put_bottle_in_fridge/fridge_base/usd/fridge_base.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_bottle_in_fridge/v2/franka_v2.pkl.gz"
    # TODO: add checker
