"""Base configuration for OGBench tasks."""

from __future__ import annotations

from dataclasses import field

from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType
from metasim.utils import configclass


@configclass
class OGBenchBaseCfg(BaseTaskCfg):
    """Base configuration for OGBench tasks.

    OGBench provides offline goal-conditioned RL environments
    with various locomotion and manipulation tasks.
    """

    # OGBench specific parameters
    dataset_name: str = ""
    single_task: bool = False  # If True, use single-task version
    task_id: int | None = None  # Task ID for single-task mode (1-5)
    use_oracle_rep: bool = False  # Use oracle goal representations
    compact_dataset: bool = False  # Use compact dataset format

    # Environment parameters
    terminate_at_goal: bool = True
    add_noise_to_goal: bool = True

    # Goal-conditioned RL parameters
    goal_conditioned: bool = True

    # Episode length varies by task
    episode_length: int = 1000

    # Reward function (OGBench uses sparse rewards)
    sparse_reward: bool = True

    # Required fields from BaseTaskCfg
    objects: list = field(default_factory=list)
    traj_filepath: str = ""  # No predefined trajectories for OGBench
    source_benchmark: BenchmarkType = BenchmarkType.OGBENCH
    reward_functions: list = field(default_factory=list)
    reward_weights: list = field(default_factory=list)

    def get_wrapper(self, num_envs: int = 1, headless: bool = True):
        """Get OGBench wrapper instance."""
        from .ogbench_wrapper import OGBenchWrapper

        return OGBenchWrapper(
            dataset_name=self.dataset_name,
            num_envs=num_envs,
            headless=headless,
            single_task=self.single_task,
            task_id=self.task_id,
            use_oracle_rep=self.use_oracle_rep,
            terminate_at_goal=self.terminate_at_goal,
            add_noise_to_goal=self.add_noise_to_goal,
            episode_length=self.episode_length,
        )
