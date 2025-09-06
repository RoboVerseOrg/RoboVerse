from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class OpenWindowTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_windowv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="window_main",
            usd_path="roboverse_data/assets/rlbench/open_window/window_main/usd/window_main.usd",
        ),
    ]
    # TODO: add checker
