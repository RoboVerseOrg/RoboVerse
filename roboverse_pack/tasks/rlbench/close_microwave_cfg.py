from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask

_OBJECTS = [
    ArticulationObjCfg(
        name="microwave_frame_resp",
        usd_path="roboverse_data/assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd",
    ),
]


@configclass
class CloseMicrowaveTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_microwavev2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class OpenMicrowaveTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_microwavev2/franka_v2.pkl.gz"
    objects = _OBJECTS
    # TODO: add checker
