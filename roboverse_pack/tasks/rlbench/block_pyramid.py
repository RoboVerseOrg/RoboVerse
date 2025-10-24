from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.block_pyramid", "block_pyramid", "franka.block_pyramid")
class BlockPyramidTask(RLBenchTask):
    max_episode_steps = 200
    traj_filepath = "roboverse_data/trajs/rlbench/block_pyramid/v2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="block_pyramid_plane",
            usd_path="roboverse_data/assets/rlbench/block_pyramid/block_pyramid_plane/usd/block_pyramid_plane.usd",
            physics=PhysicStateType.GEOM,
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block0",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block1",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block2",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block3",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block4",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_block5",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[0.0, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block0",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block1",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block2",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block3",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block4",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="block_pyramid_distractor_block5",
            physics=PhysicStateType.RIGIDBODY,
            size=[0.0375, 0.0375, 0.0375],
            color=[1.0, 0.0, 0.0],
        ),
    ]
    # TODO: add checker
