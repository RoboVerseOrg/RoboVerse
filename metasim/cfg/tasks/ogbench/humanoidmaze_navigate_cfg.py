"""HumanoidMaze navigation task from OGBench."""

from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class HumanoidMazeLargeNavigateCfg(OGBenchBaseCfg):
    """HumanoidMaze large navigation task from OGBench."""

    dataset_name: str = "humanoidmaze-large-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    # This is a goal-conditioned task
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class HumanoidMazeMediumNavigateCfg(OGBenchBaseCfg):
    """HumanoidMaze medium navigation task from OGBench."""

    dataset_name: str = "humanoidmaze-medium-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class HumanoidMazeGiantNavigateCfg(OGBenchBaseCfg):
    """HumanoidMaze giant navigation task from OGBench."""

    dataset_name: str = "humanoidmaze-giant-navigate-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False
