from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class ReachAndDragTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/reach_and_dragv2/franka_v2.pkl.gz"
    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=[0.08, 0.08, 0.08],
            color=[0.85, 0.85, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="stick",
            size=[0.0128, 0.0128, 0.36],
            color=[0.85, 0.85, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="target0",
            usd_path="roboverse_data/assets/rlbench/reach_and_drag/target0/usd/target0.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
