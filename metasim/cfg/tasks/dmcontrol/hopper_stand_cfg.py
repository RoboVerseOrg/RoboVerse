from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class HopperStandCfg(DMControlBaseCfg):
    """Hopper stand task from dm_control suite."""

    domain_name = "hopper"
    task_name = "stand"
    episode_length = 1000
    task_type = TaskType.LOCOMOTION  # Standing is a locomotion-related task
