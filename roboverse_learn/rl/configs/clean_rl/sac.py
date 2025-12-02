from typing import Optional

from metasim.utils import configclass

from roboverse_learn.rl.configs.clean_rl.base import BaseRLConfig, SimBackend


@configclass
class CleanRLSACConfig(BaseRLConfig):
    """CleanRL SAC configuration adapted for RoboVerse."""

    # Experiment
    exp_name: str = "sac"

    # Tracking / logging flags (CleanRL-style)
    track: bool = False
    wandb_project: str = "cleanRL"
    capture_video: bool = False

    # RoboVerse specific arguments
    task: str = "reach_origin"
    robot: str = "franka"
    sim: SimBackend = "mjx"
    headless: bool = False
    device: str = "cuda"

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    num_envs: int = 128
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 10
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True


__all__ = ["CleanRLSACConfig"]
