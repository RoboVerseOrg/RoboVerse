from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class GAPartNetTaskCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.GAPARTNET
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
