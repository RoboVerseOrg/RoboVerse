from metasim.constants import TaskType
from metasim.utils import configclass

from .ogbench_base import OGBenchBaseCfg


@configclass
class VisualHumanoidMazeMediumCfg(OGBenchBaseCfg):
    """Configuration for visual humanoid maze medium navigation task.

    A vision-based navigation task where a humanoid robot navigates through
    a medium-sized maze using visual feedback for bipedal locomotion.
    """

    dataset_name: str = "visual-humanoidmaze-medium-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 2000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualHumanoidMazeLargeCfg(OGBenchBaseCfg):
    """Configuration for visual humanoid maze large navigation task.

    A challenging vision-based navigation with a humanoid in a large maze,
    requiring robust visual perception and complex locomotion control.
    """

    dataset_name: str = "visual-humanoidmaze-large-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 2000
    goal_conditioned: bool = True
    single_task: bool = False


@configclass
class VisualHumanoidMazeGiantCfg(OGBenchBaseCfg):
    """Configuration for visual humanoid maze giant navigation task.

    An extremely challenging visual navigation task with a humanoid robot
    in a giant maze, testing limits of visual perception and bipedal control.
    """

    dataset_name: str = "visual-humanoidmaze-giant-v0"
    task_type = TaskType.NAVIGATION
    episode_length: int = 4000
    goal_conditioned: bool = True
    single_task: bool = False
