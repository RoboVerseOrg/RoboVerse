from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class VisualPuzzle3x3Cfg(OGBenchBaseCfg):
    """Configuration for visual 3x3 puzzle manipulation task.

    A vision-based sliding puzzle task with 3x3 grid, requiring visual
    perception to solve puzzle configurations from camera observations.
    """

    dataset_name: str = "visual-puzzle-3x3-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualPuzzle4x4Cfg(OGBenchBaseCfg):
    """Configuration for visual 4x4 puzzle manipulation task.

    A more complex vision-based puzzle with 4x4 grid demanding advanced
    visual reasoning and multi-step planning from visual inputs.
    """

    dataset_name: str = "visual-puzzle-4x4-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualPuzzle4x5Cfg(OGBenchBaseCfg):
    """Configuration for visual 4x5 puzzle manipulation task.

    A challenging vision-based rectangular puzzle requiring sophisticated
    visual perception and long-horizon planning capabilities.
    """

    dataset_name: str = "visual-puzzle-4x5-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualPuzzle4x6Cfg(OGBenchBaseCfg):
    """Configuration for visual 4x6 puzzle manipulation task.

    The most complex visual puzzle variant with 4x6 grid, testing limits
    of vision-based reasoning and optimal solution finding.
    """

    dataset_name: str = "visual-puzzle-4x6-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
