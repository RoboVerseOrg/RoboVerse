import math

from metasim.cfg.checkers import JointPosShiftChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .gapartmanip_base_metacfg import GAPartManipBaseTaskCfg


@configclass
class GAPartManipOpenBoxCfg(GAPartManipBaseTaskCfg):
    """Configuration for the GAPartManip task of 'open box' manipulation.

    This task involves manipulating a box object to open it.
    """

    decimation = 3
    episode_length = 300
    objects = [
        ArticulationObjCfg(
            name="box",
            fix_base_link=True,
            urdf_path="roboverse_data/assets/gapartmanip/Box/100189/mobility_annotation_gapartnet_joint_axis_normalized.urdf",
            scale=0.3,
        ),
    ]
    traj_filepath = "metasim/cfg/tasks/gapartmanip/example_open_box_v2.json"
    checker = JointPosShiftChecker(
        obj_name="box",
        joint_name="joint_0",
        threshold=30 / 180 * math.pi,
    )

    def reward_fn(self, states):
        """Not implemented yet."""
        return 0.0
