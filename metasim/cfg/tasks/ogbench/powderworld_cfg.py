from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class PowderworldEasyCfg(OGBenchBaseCfg):
    """Configuration for Powderworld easy difficulty task.

    A tabletop manipulation task involving powder-like materials with
    easy difficulty, requiring basic material handling skills.
    """

    dataset_name: str = "powderworld-easy-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class PowderworldMediumCfg(OGBenchBaseCfg):
    """Configuration for Powderworld medium difficulty task.

    A moderate difficulty task with powder materials requiring more precise
    control and understanding of granular material dynamics.
    """

    dataset_name: str = "powderworld-medium-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class PowderworldHardCfg(OGBenchBaseCfg):
    """Configuration for Powderworld hard difficulty task.

    A challenging manipulation task with powder materials demanding
    advanced control strategies and complex material interactions.
    """

    dataset_name: str = "powderworld-hard-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False
