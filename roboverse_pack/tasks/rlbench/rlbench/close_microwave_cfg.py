from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg

_OBJECTS = [
    ArticulationObjCfg(
        name="microwave_frame_resp",
        usd_path="roboverse_data/assets/rlbench/close_microwave/microwave_frame_resp/usd/microwave_frame_resp.usd",
    ),
]


@configclass
class CloseMicrowaveCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_microwave/v2"
    objects = _OBJECTS
    # TODO: add checker


@configclass
class OpenMicrowaveCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_microwave/v2"
    objects = _OBJECTS
    # TODO: add checker
