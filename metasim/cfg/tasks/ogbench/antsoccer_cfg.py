from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class AntSoccerArenaCfg(OGBenchBaseCfg):
    """Configuration for the Ant Soccer Arena task.

    A goal-conditioned navigation task where an ant robot plays soccer
    in an arena environment, navigating to reach target positions.
    """

    dataset_name: str = "antsoccer-arena-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class AntSoccerMediumCfg(OGBenchBaseCfg):
    """Configuration for the Ant Soccer Medium task.

    A medium-difficulty goal-conditioned navigation task where an ant robot
    plays soccer, with intermediate complexity compared to arena variant.
    """

    dataset_name: str = "antsoccer-medium-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
