from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class PushButtonsTask(RLBenchTask):
    episode_length = 500
    traj_filepath = "roboverse_data/trajs/rlbench/push_buttonsv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="push_buttons_target0",
            usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target0/usd/push_buttons_target0.usd",
        ),
        ArticulationObjCfg(
            name="push_buttons_target1",
            usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target1/usd/push_buttons_target1.usd",
        ),
        ArticulationObjCfg(
            name="push_buttons_target2",
            usd_path="roboverse_data/assets/rlbench/push_buttons/push_buttons_target2/usd/push_buttons_target2.usd",
        ),
    ]
    # TODO: add checker
