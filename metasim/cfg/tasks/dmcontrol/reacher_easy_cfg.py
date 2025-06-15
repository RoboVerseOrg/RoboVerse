from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class ReacherEasyCfg(DMControlBaseCfg):
    """Reacher easy task from dm_control suite."""

    domain_name = "reacher"
    task_name = "easy"
    episode_length = 1000
    task_type = TaskType.TABLETOP_MANIPULATION  # Reaching tasks are manipulation tasks
