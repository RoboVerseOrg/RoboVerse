from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PickAndLiftTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/pick_and_liftv2/franka_v2.pkl.gz"
    objects = [
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
    ]
    # TODO: add checker
