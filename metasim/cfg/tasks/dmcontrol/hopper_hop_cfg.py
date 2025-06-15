from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class HopperHopCfg(DMControlBaseCfg):
    domain_name: str = "hopper"
    task_name: str = "hop"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION

    # Observation: 15 dims, Action: 4 dims
