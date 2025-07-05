from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class FingerSpinCfg(DMControlBaseCfg):
    domain_name: str = "finger"
    task_name: str = "spin"
    episode_length: int = 1000
    task_type = TaskType.TABLETOP_MANIPULATION

    # Observation: 9 dims, Action: 2 dims
