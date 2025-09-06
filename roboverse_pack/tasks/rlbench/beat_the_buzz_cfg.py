from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class BeatTheBuzzTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/beat_the_buzzv2/franka_v2.pkl.gz"
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
