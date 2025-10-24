from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.stack_blocks", "stack_blocks", "franka.stack_blocks")
class StackBlocksTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="stack_blocks_target_plane",
                usd_path="roboverse_data/assets/rlbench/stack_blocks/stack_blocks_target_plane/usd/stack_blocks_target_plane.usd",
                physics=PhysicStateType.GEOM,
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_target0",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_target1",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_target2",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_target3",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 0.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor0",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 1.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor1",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 1.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor2",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 1.0, 0.0],
            ),
            PrimitiveCubeCfg(
                name="stack_blocks_distractor3",
                physics=PhysicStateType.RIGIDBODY,
                size=[0.05, 0.05, 0.05],
                color=[1.0, 1.0, 0.0],
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/stack_blocks/v2/franka_v2.pkl.gz"
    # TODO: add checker
