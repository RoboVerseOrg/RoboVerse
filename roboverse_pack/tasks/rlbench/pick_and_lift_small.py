from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.pick_and_lift_small", "pick_and_lift_small", "franka.pick_and_lift_small")
class PickAndLiftSmallTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="triangular_prism",
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/triangular_prism/usd/triangular_prism.usd",
            ),
            RigidObjCfg(
                name="star_visual",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/star_visual/usd/star_visual.usd",
            ),
            RigidObjCfg(
                name="moon_visual",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/moon_visual/usd/moon_visual.usd",
            ),
            RigidObjCfg(
                name="cylinder",
                physics=PhysicStateType.XFORM,
                usd_path="roboverse_data/assets/rlbench/pick_and_lift_small/cylinder/usd/cylinder.usd",
            ),
            PrimitiveCubeCfg(
                name="cube",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.02089, 0.02089, 0.02089],
                color=[0.0, 0.85, 1.0],
            ),
            PrimitiveSphereCfg(
                name="success_visual",
                physics=PhysicStateType.XFORM,
                color=[1.0, 0.14, 0.14],
                radius=0.04,
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/pick_and_lift_small/v2/franka_v2.pkl.gz"
    # TODO: add checker
