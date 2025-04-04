from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

from .metaworld_task_cfg import MetaworldTaskCfg


@configclass
class WindowCloseCfg(MetaworldTaskCfg):
    source_benchmark = BenchmarkType.METAWORLD
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 500
    decimation = 5
    objects = [
        ArticulationObjCfg(
            name="window",
            usd_path="data_isaaclab/assets/metaworld/window.usd",
            mjcf_path="data_isaaclab/assets_mujoco/metaworld/window.xml",
            fix_base_link=True,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/metaworld/window_close_v2.pkl"  # fmt: skip

    checker = JointPosChecker(
        obj_name="window",
        joint_name="window_slide",
        mode="le",
        radian_threshold=0.04,
    )
