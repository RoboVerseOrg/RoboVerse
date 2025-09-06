from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class OpenGrillTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_grillv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="grill",
            usd_path="roboverse_data/assets/rlbench/open_grill/grill/usd/grill.usd",
        ),
    ]
    # TODO: add checker
