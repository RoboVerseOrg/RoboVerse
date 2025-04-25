import math

from metasim.cfg.checkers import JointPosShiftChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .gapartmanip_base_metacfg import GAPartManipBaseTaskCfg


@configclass
class GAPartManipOpenToiletCfg(GAPartManipBaseTaskCfg):
    """Configuration for the GAPartManip task of 'open toilet' manipulation.

    This task involves manipulating a toilet object to open it.
    """

    decimation = 3
    episode_length = 300
    objects = [
        ArticulationObjCfg(
            name="toilet",
            fix_base_link=True,
            urdf_path="roboverse_data/assets/gapartmanip/Toilet/102630/mobility_annotation_gapartnet.urdf",
            scale=0.3,
            enabled_self_collisions=False,
        ),
    ]
    traj_filepath = "metasim/cfg/tasks/gapartmanip/example_open_toilet_v2.json"
    checker = JointPosShiftChecker(
        obj_name="toilet",
        joint_name="joint_0",
        threshold=30 / 180 * math.pi,
    )

    def reward_fn(self, states):
        """Not implemented yet."""
        return 0.0
