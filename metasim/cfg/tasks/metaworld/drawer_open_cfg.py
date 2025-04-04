from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.utils import configclass

from .metaworld_task_cfg import MetaworldTaskCfg


@configclass
class DrawerOpenCfg(MetaworldTaskCfg):
    episode_length = 500
    decimation = 5
    objects = [
        ArticulationObjCfg(
            name="drawer",
            usd_path="roboverse_data/assets/metaworld/drawer_open/drawer/usd/drawer.usd",
            mjcf_path="data_isaaclab/assets_mujoco/metaworld/drawer.xml",
            fix_base_link=True,
        ),
    ]

    traj_filepath = "data_isaaclab/source_data/metaworld/drawer_open_v2.pkl"

    checker = JointPosChecker(
        obj_name="drawer",
        joint_name="goal_slidey",
        mode="le",
        radian_threshold=-0.1,
    )
