from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class VisualCubeSingleCfg(OGBenchBaseCfg):
    """Configuration for visual single cube manipulation task.

    A vision-based manipulation task with a single cube, requiring
    visual perception to achieve precise object positioning goals.
    """

    dataset_name: str = "visual-cube-single-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 200
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualCubeDoubleCfg(OGBenchBaseCfg):
    """Configuration for visual double cube manipulation task.

    A vision-based task manipulating two cubes, demanding visual
    coordination and spatial reasoning from camera observations.
    """

    dataset_name: str = "visual-cube-double-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 500
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualCubeTripleCfg(OGBenchBaseCfg):
    """Configuration for visual triple cube manipulation task.

    A complex vision-based manipulation task with three cubes requiring
    advanced visual perception and multi-object coordination strategies.
    """

    dataset_name: str = "visual-cube-triple-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualCubeQuadrupleCfg(OGBenchBaseCfg):
    """Configuration for visual quadruple cube manipulation task.

    The most challenging visual cube task with four objects, testing
    limits of vision-based multi-object manipulation and planning.
    """

    dataset_name: str = "visual-cube-quadruple-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 1000
    goal_conditioned: bool = True
    single_task: bool = False
