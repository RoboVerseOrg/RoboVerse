from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

_OBJECTS = [
    ArticulationObjCfg(
        name="fridge_base",
        usd_path="roboverse_data/assets/rlbench/close_fridge/fridge_base/usd/fridge_base.usd",
    ),
]


@configclass
class CloseFridgeTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_fridgev2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class OpenFridgeTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_fridgev2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
