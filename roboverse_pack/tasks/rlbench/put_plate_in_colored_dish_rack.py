from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task(
    "rlbench.put_plate_in_colored_dish_rack", "put_plate_in_colored_dish_rack", "franka.put_plate_in_colored_dish_rack"
)
class PutPlateInColoredDishRackTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="plate_visual",
                usd_path="roboverse_data/assets/rlbench/put_plate_in_colored_dish_rack/plate_visual/usd/plate_visual.usd",
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="dish_rack",
                usd_path="roboverse_data/assets/rlbench/put_plate_in_colored_dish_rack/dish_rack/usd/dish_rack.usd",
                physics=PhysicStateType.GEOM,
            ),
            RigidObjCfg(
                name="plate_stand",
                usd_path="roboverse_data/assets/rlbench/put_plate_in_colored_dish_rack/plate_stand/usd/plate_stand.usd",
                physics=PhysicStateType.GEOM,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/put_plate_in_colored_dish_rack/v2/franka_v2.pkl.gz"
    # TODO: add checker
