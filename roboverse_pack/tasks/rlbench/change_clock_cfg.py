from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class ChangeClockTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/change_clockv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="clock",
            usd_path="roboverse_data/assets/rlbench/change_clock/clock/usd/clock.usd",
        ),
    ]
    # TODO: add checker
