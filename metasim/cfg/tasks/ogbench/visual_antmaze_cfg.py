from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class VisualAntMazeMediumCfg(OGBenchBaseCfg):
    """Configuration for visual AntMaze medium navigation task.

    A vision-based navigation task where an ant robot navigates through
    a medium-sized maze using visual observations to reach goals.
    """

    dataset_name: str = "visual-antmaze-medium-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualAntMazeLargeCfg(OGBenchBaseCfg):
    """Configuration for visual AntMaze large navigation task.

    A challenging vision-based navigation task in a large maze environment
    requiring visual perception and efficient exploration strategies.
    """

    dataset_name: str = "visual-antmaze-large-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualAntMazeGiantCfg(OGBenchBaseCfg):
    """Configuration for visual AntMaze giant navigation task.

    An extremely challenging visual navigation task in a giant maze
    demanding robust visual perception and long-horizon planning.
    """

    dataset_name: str = "visual-antmaze-giant-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualAntMazeTeleportCfg(OGBenchBaseCfg):
    """Configuration for visual AntMaze teleport navigation task.

    A visual navigation task with teleportation mechanics, combining
    vision-based control with strategic portal usage for efficient travel.
    """

    dataset_name: str = "visual-antmaze-teleport-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
