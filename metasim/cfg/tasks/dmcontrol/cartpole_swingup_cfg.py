from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class CartpoleSwingupCfg(DMControlBaseCfg):
    domain_name: str = "cartpole"
    task_name: str = "swingup"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION

    # Observation: 5 dims, Action: 1 dim
