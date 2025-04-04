from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class BeatTheBuzzCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/beat_the_buzz/v2"
    objects = [
        RigidObjCfg(
            name="wand",
            usd_path="roboverse_data/assets/rlbench/beat_the_buzz/wand/usd/wand.usd",
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
            name="Cuboid",
            usd_path="roboverse_data/assets/rlbench/beat_the_buzz/Cuboid/usd/Cuboid.usd",
            physics=PhysicStateType.GEOM,
        ),
    ]
    # TODO: add checker
