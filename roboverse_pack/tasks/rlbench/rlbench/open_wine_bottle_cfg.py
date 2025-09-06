from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class OpenWineBottleCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_wine_bottle/v2"
    objects = [
        ArticulationObjCfg(
            name="bottle",
            usd_path="roboverse_data/assets/rlbench/open_wine_bottle/bottle/usd/bottle.usd",
        ),
    ]
    # TODO: add checker
