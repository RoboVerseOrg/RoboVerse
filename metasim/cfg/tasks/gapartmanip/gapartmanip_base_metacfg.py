from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class GAPartManipBaseTaskCfg(BaseTaskCfg):
    """Base configuration for GAPartManip tasks."""

    source_benchmark = BenchmarkType.GAPARTMANIP
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
