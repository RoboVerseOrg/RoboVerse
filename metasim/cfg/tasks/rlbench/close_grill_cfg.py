from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class CloseGrillCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/close_grill/v2"
    objects = [
        ArticulationObjCfg(
            name="grill",
            usd_path="roboverse_data/assets/rlbench/close_grill/grill/usd/grill.usd",
        ),
    ]
    # TODO: add checker
