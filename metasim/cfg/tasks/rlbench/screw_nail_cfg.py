from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class ScrewNailCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/screw_nail/v2"
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
