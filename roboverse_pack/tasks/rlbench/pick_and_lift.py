from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.pick_and_lift", "pick_and_lift", "franka.pick_and_lift")
class PickAndLiftTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="pick_and_lift_target",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor0",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[0.0, 1.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor1",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 1.0, 1.0],
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
    traj_filepath = "roboverse_data/trajs/rlbench/pick_and_lift/v2/franka_v2.pkl.gz"
    # TODO: add checker
