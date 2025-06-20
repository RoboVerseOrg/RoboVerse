from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class Puzzle3x3Cfg(OGBenchBaseCfg):
    """Configuration for 3x3 puzzle manipulation task.

    A tabletop task involving solving a 3x3 sliding puzzle through
    sequential piece movements to achieve target configurations.
    """

    dataset_name: str = "puzzle-3x3-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class Puzzle4x4Cfg(OGBenchBaseCfg):
    """Configuration for 4x4 puzzle manipulation task.

    A more complex sliding puzzle task with 4x4 grid requiring
    advanced planning and piece coordination strategies.
    """

    dataset_name: str = "puzzle-4x4-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class Puzzle4x5Cfg(OGBenchBaseCfg):
    """Configuration for 4x5 puzzle manipulation task.

    A challenging rectangular puzzle with 4x5 grid configuration
    demanding sophisticated multi-step planning abilities.
    """

    dataset_name: str = "puzzle-4x5-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class Puzzle4x6Cfg(OGBenchBaseCfg):
    """Configuration for 4x6 puzzle manipulation task.

    The most complex puzzle variant with 4x6 grid requiring
    long-horizon planning and optimal piece movement sequences.
    """

    dataset_name: str = "puzzle-4x6-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
