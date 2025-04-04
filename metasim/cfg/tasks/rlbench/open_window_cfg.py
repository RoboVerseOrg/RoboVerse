from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class OpenWindowCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_window/v2"
    objects = [
        ArticulationObjCfg(
            name="window_main",
            usd_path="roboverse_data/assets/rlbench/open_window/window_main/usd/window_main.usd",
        ),
    ]
    # TODO: add checker
