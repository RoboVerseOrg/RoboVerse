import math

from metasim.example.example_pack.tasks.checkers import JointPosChecker
from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils import configclass

from .rl_bench import RLBenchTask


@configclass
class CloseBoxTask(RLBenchTask):
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
    traj_filepath = "metasim/data/quick_start/trajs/rlbench/close_boxv2/franka_v2.pkl.gz"
    checker = JointPosChecker(
        obj_name="box_base",
        joint_name="box_joint",
        mode="le",
        radian_threshold=-14 / 180 * math.pi,
    )
