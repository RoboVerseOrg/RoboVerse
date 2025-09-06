from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class OpenWashingMachineCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_washing_machine/v2"
    objects = [
        ArticulationObjCfg(
            name="washer",
            usd_path="roboverse_data/assets/rlbench/open_washing_machine/washer/usd/washer.usd",
        ),
    ]
    # TODO: add checker
