from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger as log

# from metasim.cfg.randomization import RandomizationCfg


@dataclass
class Args:
    task: str
    """Task name"""
    # random: RandomizationCfg = field(default_factory=RandomizationCfg)
    """Domain randomization options"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaacsim"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""

    def __post_init__(self):
        # if self.random.table and not self.table:
        #     log.warning("Cannot enable table randomization without a table, disabling table randomization")
        #     self.random.table = False
        log.info(f"Args: {self}")
