from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class VisualSceneCfg(OGBenchBaseCfg):
    """Configuration for visual scene manipulation task.

    A vision-based complex manipulation task with multiple objects in a scene,
    requiring visual perception to arrange objects according to goal images.
    """

    dataset_name: str = "visual-scene-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 750
    goal_conditioned: bool = True
    single_task: bool = False
