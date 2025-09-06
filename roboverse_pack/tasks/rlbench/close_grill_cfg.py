from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class CloseGrillTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_grillv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="grill",
            usd_path="roboverse_data/assets/rlbench/close_grill/grill/usd/grill.usd",
        ),
    ]
    # TODO: add checker
