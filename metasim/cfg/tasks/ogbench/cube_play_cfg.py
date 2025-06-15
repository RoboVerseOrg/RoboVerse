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
