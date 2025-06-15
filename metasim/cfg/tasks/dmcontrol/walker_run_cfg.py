from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class WalkerRunCfg(DMControlBaseCfg):
    domain_name: str = "walker"
    task_name: str = "run"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION

    # Observation: 24 dims, Action: 6 dims
