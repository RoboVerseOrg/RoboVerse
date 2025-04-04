import math

from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.cfg.tasks import BaseTaskCfg
from metasim.utils import configclass


@configclass
class FetchCloseBoxCfg(BaseTaskCfg):
    episode_length = 250
    objects = [
        ArticulationObjCfg(
            name="box_base",
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]
    traj_filepath = "metasim/cfg/tasks/fetch/fetch_example_v2.json"
    checker = JointPosChecker(
        obj_name="box_base",
        joint_name="box_joint",
        mode="le",
        radian_threshold=-14 / 180 * math.pi,
    )
