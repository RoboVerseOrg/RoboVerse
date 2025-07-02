from metasim.constants import TaskType
from metasim.utils import configclass

from .dmcontrol_base import DMControlBaseCfg


@configclass
class AcrobotSwingupCfg(DMControlBaseCfg):
    domain_name: str = "acrobot"
    task_name: str = "swingup"
    episode_length: int = 1000
    task_type = TaskType.LOCOMOTION
