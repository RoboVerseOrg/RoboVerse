from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class FingerTurnHardCfg(DMControlBaseCfg):
    domain_name: str = "finger"
    task_name: str = "turn_hard"
    episode_length: int = 1000
    task_type = TaskType.TABLETOP_MANIPULATION

    # Observation: 12 dims, Action: 2 dims
