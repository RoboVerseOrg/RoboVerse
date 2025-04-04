from metasim.cfg.checkers import PositionShiftChecker
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class PushRedBlockRightCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.CALVIN
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 600
    checker = PositionShiftChecker(
        obj_name="block_red",
        distance=0.065,
        axis="x",
    )
