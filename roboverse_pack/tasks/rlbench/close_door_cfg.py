from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

OBJECTS = [
    ArticulationObjCfg(
        name="door_frame",
        usd_path="roboverse_data/assets/rlbench/close_door/door_frame/usd/door_frame.usd",
    ),
]


@configclass
class CloseDoorTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_doorv2/franka_v2.pkl.gz"
    objects = OBJECTS
    # TODO: add checker


@configclass
class OpenDoorTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_doorv2/franka_v2.pkl.gz"
    objects = OBJECTS
    # TODO: add checker
