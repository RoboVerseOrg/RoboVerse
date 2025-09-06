from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class OpenBoxCfg(RLBenchTaskCfg):
    episode_length = 250
    traj_filepath = "roboverse_data/trajs/rlbench/open_box/v2"
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
