from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class PendulumSwingupCfg(DMControlBaseCfg):
    domain_name: str = "pendulum"
    task_name: str = "swingup"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION

    # Observation: 3 dims, Action: 1 dim
