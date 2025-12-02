from typing import Tuple

from metasim.utils import configclass

from roboverse_learn.rl.configs.base import BaseRLConfig, SimBackend


@configclass
class RslRlPPOConfig(BaseRLConfig):
    """RSL-RL PPO-specific configuration."""

    # Override base experiment defaults
    exp_name: str = "rsl_rl_ppo"

    # RSL-RL style runner settings
    num_steps_per_env: int = 24
    empirical_normalization: bool = False

    # Policy Network
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0

    # PPO Algorithm
    learning_rate: float = 1e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True

    # Logging
    wandb_project: str = "rsl_rl_ppo"


__all__ = ["RslRlPPOConfig"]
