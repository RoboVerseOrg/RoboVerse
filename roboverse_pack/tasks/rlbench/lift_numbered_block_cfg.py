from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class LiftNumberedBlockTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/lift_numbered_blockv2/franka_v2.pkl.gz"
    objects = [
        RigidObjCfg(
            name="block1",
            usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block1/usd/block1.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="block2",
            usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block2/usd/block2.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="block3",
            usd_path="roboverse_data/assets/rlbench/lift_numbered_block/block3/usd/block3.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
