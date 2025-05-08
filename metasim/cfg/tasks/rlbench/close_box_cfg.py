import math

from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rlbench_task_cfg import RLBenchTaskCfg


@configclass
class CloseBoxCfg(RLBenchTaskCfg):
    episode_length = 250
    objects = [
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="metasim/data/quick_start/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]
    traj_filepath = "metasim/data/quick_start/trajs/rlbench/close_box/v2"
    checker = JointPosChecker(
        obj_name="box_base",
        joint_name="box_joint",
        mode="le",
        radian_threshold=-14 / 180 * math.pi,
    )
