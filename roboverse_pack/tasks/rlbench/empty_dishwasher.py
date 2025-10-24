from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.empty_dishwasher", "empty_dishwasher", "franka.empty_dishwasher")
class EmptyDishwasherTask(RLBenchTask):
    max_episode_steps = 600
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="dishwasher",
                usd_path="roboverse_data/assets/rlbench/empty_dishwasher/dishwasher/usd/dishwasher.usd",
            ),
            RigidObjCfg(
                name="dishwasher_plate_visual",
                usd_path="roboverse_data/assets/rlbench/empty_dishwasher/dishwasher_plate_visual/usd/dishwasher_plate_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/empty_dishwasher/v2/franka_v2.pkl.gz"
    # TODO: add checker
