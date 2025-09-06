from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class StackBlocksCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/stack_blocks/v2"
    objects = [
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
    ]
    # TODO: add checker
