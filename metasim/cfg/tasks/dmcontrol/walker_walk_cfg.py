from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class WalkerWalkCfg(DMControlBaseCfg):
    """Walker walk task from dm_control suite."""

    domain_name = "walker"
    task_name = "walk"
    episode_length = 1000
    task_type = TaskType.LOCOMOTION

    move_speed = 1.0
