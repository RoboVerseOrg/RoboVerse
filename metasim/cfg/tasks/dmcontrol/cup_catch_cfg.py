from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class CupCatchCfg(DMControlBaseCfg):
    domain_name: str = "ball_in_cup"
    task_name: str = "catch"
    episode_length: int = 1000
    task_type = TaskType.TABLETOP_MANIPULATION

    # Observation: 8 dims, Action: 2 dims
