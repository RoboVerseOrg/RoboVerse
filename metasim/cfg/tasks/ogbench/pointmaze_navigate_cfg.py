from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class PointMazeMediumNavigateCfg(OGBenchBaseCfg):
    """Configuration for PointMaze medium navigation task.

    A goal-conditioned navigation task where a point mass navigates through
    a medium-sized maze environment to reach target locations.
    """

    dataset_name: str = "pointmaze-medium-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class PointMazeLargeNavigateCfg(OGBenchBaseCfg):
    """Configuration for PointMaze large navigation task.

    A challenging navigation task with a point mass in a large maze,
    requiring efficient path planning to reach distant goals.
    """

    dataset_name: str = "pointmaze-large-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class PointMazeGiantNavigateCfg(OGBenchBaseCfg):
    """Configuration for PointMaze giant navigation task.

    An extremely challenging navigation task in a giant maze environment,
    testing long-horizon planning and exploration capabilities.
    """

    dataset_name: str = "pointmaze-giant-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class PointMazeTeleportNavigateCfg(OGBenchBaseCfg):
    """Configuration for PointMaze teleport navigation task.

    A navigation task with teleportation mechanics, where the point mass
    can use special portals to instantly travel between maze sections.
    """

    dataset_name: str = "pointmaze-teleport-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
