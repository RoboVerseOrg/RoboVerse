from metasim.cfg.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class InsertOntoSquarePegCfg(RLBenchTaskCfg):
    episode_length = 600
    traj_filepath = "roboverse_data/trajs/rlbench/insert_onto_square_peg/v2"
    objects = [
        RigidObjCfg(
            name="square_ring",
            usd_path="roboverse_data/assets/rlbench/insert_onto_square_peg/square_ring/usd/square_ring.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="pillar0",
            size=[0.025, 0.025, 0.12],
            physics=PhysicStateType.GEOM,
            color=[1.0, 0.0, 1.0],
        ),
        PrimitiveCubeCfg(
            name="pillar1",
            physics=PhysicStateType.GEOM,
            size=[0.025, 0.025, 0.12],
            color=[0.5, 0.5, 0.0],
        ),
        PrimitiveCubeCfg(
            name="pillar2",
            physics=PhysicStateType.GEOM,
            size=[0.025, 0.025, 0.12],
            color=[1.0, 0.0, 0.0],
        ),
        PrimitiveCubeCfg(
            name="square_base",
            physics=PhysicStateType.GEOM,
            size=[0.4, 0.1, 0.02],
            color=[0.49, 0.38, 0.29],
        ),
    ]
    # TODO: add checker
