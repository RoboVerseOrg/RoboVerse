from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class CartpoleBalanceSparseCfg(DMControlBaseCfg):
    domain_name: str = "cartpole"
    task_name: str = "balance_sparse"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION
