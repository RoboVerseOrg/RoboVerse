from typing import Literal

from metasim.utils import configclass

from .ppo_base import BasePPOConfig


@configclass
class RslRlPPOConfig(BasePPOConfig):
    """RSL-RL PPO-specific configuration."""

    # Override base experiment defaults
    exp_name: str = "rsl_rl_ppo"

    # Override base task/env defaults for RSL-RL PPO
    sim: Literal[
        "isaacgym",
        "isaacsim",
        "isaaclab",
        "mujoco",
        "genesis",
        "mjx",
    ] = "isaacsim"
    num_envs: int = 512
    headless: bool = False

    # Logging
    wandb_project: str = "rsl_rl_ppo"
