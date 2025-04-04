from metasim.cfg.checkers import JointPosChecker
from metasim.cfg.objects import ArticulationObjCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

from .metaworld_task_cfg import MetaworldTaskCfg


@configclass
class DoorCloseCfg(MetaworldTaskCfg):
    source_benchmark = BenchmarkType.METAWORLD
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 500
    decimation = 5
    objects = [
        ArticulationObjCfg(
            name="door",
            usd_path="data_isaaclab/assets/metaworld/door.usd",
            mjcf_path="data_isaaclab/assets_mujoco/metaworld/door.xml",
            fix_base_link=True,
        ),
    ]
    traj_filepath = "data_isaaclab/source_data/metaworld/door_close_v2.pkl"  # fmt: skip

    checker = JointPosChecker(
        obj_name="door",
        joint_name="doorjoint",
        mode="ge",
        radian_threshold=-0.15,
    )
