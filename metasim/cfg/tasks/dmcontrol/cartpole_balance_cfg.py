from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class CartpoleBalanceCfg(DMControlBaseCfg):
    """Cartpole balance task from dm_control suite."""

    domain_name = "cartpole"
    task_name = "balance"
    episode_length = 1000
    task_type = TaskType.LOCOMOTION  # Balancing tasks can be considered a form of locomotion
