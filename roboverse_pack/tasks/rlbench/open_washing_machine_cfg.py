from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class OpenWashingMachineTask(RLBenchTask):
    episode_length = 200
    traj_filepath = "roboverse_data/trajs/rlbench/open_washing_machinev2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="washer",
            usd_path="roboverse_data/assets/rlbench/open_washing_machine/washer/usd/washer.usd",
        ),
    ]
    # TODO: add checker
