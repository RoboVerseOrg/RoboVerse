from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger as log


@dataclass
class Args:
    task: str
    """Task name"""
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

    # Domain Randomization options
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"""
    scene_mode: Literal[0, 1, 2, 3] = 0
    """Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    def __post_init__(self):
        log.info(f"Args: {self}")
