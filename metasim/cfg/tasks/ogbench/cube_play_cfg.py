"""Cube manipulation tasks from OGBench."""

from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class CubeDoublePlayCfg(OGBenchBaseCfg):
    """Cube double play task from OGBench."""

    dataset_name: str = "cube-double-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000

    # This is a goal-conditioned task
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeDoublePlaySingleTaskCfg(OGBenchBaseCfg):
    """Cube double play single-task variant (default task 2)."""

    dataset_name: str = "cube-double-play-singletask-task2-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000

    # Single-task version
    goal_conditioned: bool = False
    single_task: bool = True
    task_id: int = 2  # Default task for cube


@configclass
class CubeTriplePlayCfg(OGBenchBaseCfg):
    """Cube triple play task from OGBench."""

    dataset_name: str = "cube-triple-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeQuadruplePlayCfg(OGBenchBaseCfg):
    """Cube quadruple play task from OGBench."""

    dataset_name: str = "cube-quadruple-play-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000

    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeSingleCfg(OGBenchBaseCfg):
    """Configuration for single cube manipulation task.

    A tabletop manipulation task involving a single cube object with
    goal-conditioned objectives requiring precise manipulation skills.
    """

    dataset_name: str = "cube-single-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 200
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeDoubleCfg(OGBenchBaseCfg):
    """Configuration for double cube manipulation task.

    A tabletop manipulation task involving two cube objects requiring
    coordinated manipulation and spatial reasoning abilities.
    """

    dataset_name: str = "cube-double-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeTripleCfg(OGBenchBaseCfg):
    """Configuration for triple cube manipulation task.

    A complex tabletop manipulation task with three cube objects requiring
    advanced planning and multi-object coordination strategies.
    """

    dataset_name: str = "cube-triple-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class CubeQuadrupleCfg(OGBenchBaseCfg):
    """Configuration for quadruple cube manipulation task.

    The most complex cube manipulation task with four objects, demanding
    sophisticated multi-object manipulation and long-horizon planning.
    """

    dataset_name: str = "cube-quadruple-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
