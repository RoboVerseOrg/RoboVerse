from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class ChangeClockCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/change_clock/v2"
    objects = [
        ArticulationObjCfg(
            name="clock",
            usd_path="roboverse_data/assets/rlbench/change_clock/clock/usd/clock.usd",
        ),
    ]
    # TODO: add checker
