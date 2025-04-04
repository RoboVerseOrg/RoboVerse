from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class ArnoldTaskCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.ARNOLD
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = False
