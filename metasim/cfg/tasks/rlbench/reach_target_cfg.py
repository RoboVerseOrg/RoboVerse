from metasim.cfg.objects import PrimitiveSphereCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class ReachTargetCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/reach_target/v2"
    objects = [
        PrimitiveSphereCfg(
            name="target",
            radius=0.025,
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.XFORM,
        ),
        PrimitiveSphereCfg(
            name="distractor0",
            radius=0.025,
            color=[1.0, 0.0, 0.5],
            physics=PhysicStateType.XFORM,
        ),
        PrimitiveSphereCfg(
            name="distractor1",
            radius=0.025,
            color=[1.0, 1.0, 0.0],
            physics=PhysicStateType.XFORM,
        ),
    ]
    # TODO: add checker
