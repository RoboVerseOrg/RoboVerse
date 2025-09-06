from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

_OBJECTS = [
    ArticulationObjCfg(
        name="drawer_frame",
        usd_path="roboverse_data/assets/rlbench/close_drawer/drawer_frame/usd/drawer_frame.usd",
    ),
]


@configclass
class CloseDrawerTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_drawerv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class OpenDrawerTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_drawerv2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
