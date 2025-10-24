from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task(
    "rlbench.place_shape_in_shape_sorter", "place_shape_in_shape_sorter", "franka.place_shape_in_shape_sorter"
)
class PlaceShapeInShapeSorterTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="shape_sorter",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/rlbench/place_shape_in_shape_sorter/shape_sorter/usd/shape_sorter.usd",
            ),
            RigidObjCfg(
                name="triangular_prism",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd",  # reuse same asset
            ),
            RigidObjCfg(
                name="star_visual",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd",  # reuse same asset
            ),
            RigidObjCfg(
                name="moon_visual",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd",  # reuse same asset
            ),
            RigidObjCfg(
                name="cylinder",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd",  # reuse same asset
            ),
            PrimitiveCubeCfg(
                name="cube",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.02089, 0.02089, 0.02089],
                color=[0.0, 0.85, 1.0],
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/place_shape_in_shape_sorter/v2/franka_v2.pkl.gz"
    # TODO: add checker
