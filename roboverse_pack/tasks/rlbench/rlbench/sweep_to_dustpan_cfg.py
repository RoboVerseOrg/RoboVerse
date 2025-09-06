from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_DIRTS = [
    PrimitiveCubeCfg(
        name=f"dirt{i}",
        size=[0.01, 0.01, 0.01],
        color=[0.46, 0.31, 0.31],
        physics=PhysicStateType.RIGIDBODY,
    )
    for i in range(5)
]


@configclass
class SweepToDustpanCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/sweep_to_dustpan/v2"
    objects = [
        RigidObjCfg(
            name="Dustpan_4",
            usd_path="roboverse_data/assets/rlbench/sweep_to_dustpan/Dustpan_4/usd/Dustpan_4.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="sweep_to_dustpan_broom_visual",
            usd_path="roboverse_data/assets/rlbench/sweep_to_dustpan/sweep_to_dustpan_broom_visual/usd/sweep_to_dustpan_broom_visual.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="broom_holder",
            size=[0.1, 0.1, 0.5],
            color=[0.35, 0.48, 0.54],
            physics=PhysicStateType.GEOM,
        ),
    ] + _DIRTS
    # TODO: add checker
