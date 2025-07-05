from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class SceneCfg(OGBenchBaseCfg):
    """Configuration for scene manipulation task.

    A complex tabletop manipulation task involving multiple objects in
    a scene that must be arranged according to goal configurations.
    """

    dataset_name: str = "scene-v0"
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length: int = 750
    goal_conditioned: bool = True
    single_task: bool = False
