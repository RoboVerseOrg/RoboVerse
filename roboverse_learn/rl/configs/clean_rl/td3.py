from typing import Optional

from metasim.utils import configclass

from roboverse_learn.rl.configs.clean_rl.base import BaseRLConfig, SimBackend


@configclass
class CleanRLTD3Config(BaseRLConfig):
    """CleanRL TD3 configuration adapted for RoboVerse."""

    # Experiment
    exp_name: str = "td3"

    # Tracking / logging flags (CleanRL-style)
    track: bool = False
    wandb_project: str = "cleanRL"
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # RoboVerse specific arguments
    task: str = "reach_origin"
    robot: str = "franka"
    sim: SimBackend = "mjx"
    headless: bool = False
    device: str = "cuda"

    # Algorithm specific arguments
    total_timesteps: int = 10000
    learning_rate: float = 3e-4
    num_envs: int = 128
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    learning_starts: int = 10
    policy_frequency: int = 2
    noise_clip: float = 0.5


__all__ = ["CleanRLTD3Config"]
