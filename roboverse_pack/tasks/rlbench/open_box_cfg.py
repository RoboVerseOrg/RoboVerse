from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class OpenBoxTask(RLBenchTask):
    episode_length = 250
    traj_filepath = "roboverse_data/trajs/rlbench/open_boxv2/franka_v2.pkl.gz"
    objects = [
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]
    # TODO: add checker
