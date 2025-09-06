from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class OpenWineBottleTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_wine_bottlev2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="bottle",
            usd_path="roboverse_data/assets/rlbench/open_wine_bottle/bottle/usd/bottle.usd",
        ),
    ]
    # TODO: add checker
