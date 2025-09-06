from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PushButtonTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/push_buttonv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="push_button_target",
            usd_path="roboverse_data/assets/rlbench/push_button/push_button_target/usd/push_button_target.usd",
        ),
    ]
    # TODO: add checker
