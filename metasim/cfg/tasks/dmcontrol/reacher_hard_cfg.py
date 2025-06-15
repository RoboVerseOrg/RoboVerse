from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class ReacherHardCfg(DMControlBaseCfg):
    domain_name: str = "reacher"
    task_name: str = "hard"
    episode_length: int = 1000
    task_type = TaskType.TABLETOP_MANIPULATION

    # Observation: 6 dims, Action: 2 dims
