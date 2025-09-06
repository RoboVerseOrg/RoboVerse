from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class ScrewNailTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/screw_nailv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="block",
            usd_path="roboverse_data/assets/rlbench/screw_nail/block/usd/block.usd",
        ),
        RigidObjCfg(
            name="screw_driver",
            usd_path="roboverse_data/assets/rlbench/screw_nail/screw_driver/usd/screw_driver.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    # TODO: add checker
