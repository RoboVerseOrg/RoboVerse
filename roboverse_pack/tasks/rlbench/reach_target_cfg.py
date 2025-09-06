from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveSphereCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class ReachTargetTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/reach_targetv2/franka_v2.pkl.gz"
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
