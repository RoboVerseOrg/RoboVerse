from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class ManiskillTaskCfg(BaseTaskCfg):
    """The base class for Maniskill tasks."""

    source_benchmark = BenchmarkType.MANISKILL
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
