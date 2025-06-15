"""AntMaze navigation task from OGBench."""

from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class AntMazeLargeNavigateCfg(OGBenchBaseCfg):
    """AntMaze large navigation task from OGBench."""

    dataset_name: str = "antmaze-large-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    # This is a goal-conditioned task
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class AntMazeLargeNavigateSingleTaskCfg(OGBenchBaseCfg):
    """AntMaze large navigation single-task variant."""

    dataset_name: str = "antmaze-large-navigate-singletask-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    # Single-task version
    goal_conditioned: bool = False
    single_task: bool = True
    task_id: int = 1  # Default task


@configclass
class AntMazeMediumNavigateCfg(OGBenchBaseCfg):
    """AntMaze medium navigation task from OGBench."""

    dataset_name: str = "antmaze-medium-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class AntMazeGiantNavigateCfg(OGBenchBaseCfg):
    """AntMaze giant navigation task from OGBench."""

    dataset_name: str = "antmaze-giant-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False
