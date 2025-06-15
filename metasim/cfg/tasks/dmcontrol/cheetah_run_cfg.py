from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class CheetahRunCfg(DMControlBaseCfg):
    """Cheetah run task from dm_control suite."""

    domain_name = "cheetah"
    task_name = "run"
    episode_length = 1000
    task_type = TaskType.LOCOMOTION
