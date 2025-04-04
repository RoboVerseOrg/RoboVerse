"""Sub-module containing the task configuration for the UH1 task."""

from metasim.cfg.tasks import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass


@configclass
class UH1TaskCfg(BaseTaskCfg):
    """UH1 task."""

    source_benchmark = BenchmarkType.UH1
    task_type = TaskType.LOCOMOTION
    episode_length = 250
    objects = []


@configclass
class MabaoguoCfg(UH1TaskCfg):
    """Mabaoguo task."""

    traj_filepath = "data_isaaclab/source_data/humanoid/maobaoguo_traj_v2.pkl"
