import math

import torch

from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .gapartnet_task_cfg import GAPartNetTaskCfg


@configclass
class GapartnetCloseDrawerCfg(GAPartNetTaskCfg):
    episode_length = 250
    objects = [
        ArticulationObjCfg(
            name="cabinet",
            fix_base_link=True,
            urdf_path="roboverse_data/assets/gapartnet/45661/mobility_annotation_gapartnet.urdf",
            scale=0.4,
        ),
    ]
    traj_filepath = "metasim/cfg/tasks/gapartnet/example_v2.json"
    checker = JointPosChecker(
        obj_name="cabinet",
        joint_name="joint_0",
        mode="ge",
        radian_threshold=30 / 180 * math.pi,
    )

    def reward_fn(self, states):
        """Hardcode current reach-origin reward function."""
        ee_poses = torch.stack([state["metasim_body_panda_hand"]["pos"] for state in states])
        distance = torch.norm(ee_poses, dim=-1)
        return -distance
