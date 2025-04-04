from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class PushButtonCfg(RLBenchTaskCfg):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/push_button/v2"
    objects = [
        ArticulationObjCfg(
            name="push_button_target",
            usd_path="roboverse_data/assets/rlbench/push_button/push_button_target/usd/push_button_target.usd",
        ),
    ]
    # TODO: add checker
