from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.get_ice_from_fridge", "get_ice_from_fridge", "franka.get_ice_from_fridge")
class GetIceFromFridgeTask(RLBenchTask):
    max_episode_steps = 600
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="fridge_base",
                usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/fridge_base/usd/fridge_base.usd",
            ),
            RigidObjCfg(
                name="cup_visual",
                usd_path="roboverse_data/assets/rlbench/get_ice_from_fridge/cup_visual/usd/cup_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/get_ice_from_fridge/v2/franka_v2.pkl.gz"
    # TODO: add checker
