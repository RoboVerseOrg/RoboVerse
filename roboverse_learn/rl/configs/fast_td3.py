from __future__ import annotations

from typing import Literal, Optional, Sequence

from metasim.utils import configclass

SimBackend = Literal[
    "isaacgym",
    "isaacsim",
    "isaaclab",
    "mujoco",
    "genesis",
    "mjx",
]


@configclass
class FastTD3Config:
    """FastTD3 training configuration (Python equivalent of YAML configs)."""

    # Experiment
    exp_name: str = "get_started_fttd3"
    seed: int = 1
    torch_deterministic: bool = True

    # Device
    cuda: bool = True
    device: str = "cuda:0"
    device_rank: int = 0

    # Environment
    sim: SimBackend = "isaacgym"
    robots: Sequence[str] = ("h1",)
    task: str = "walk"
    decimation: int = 10
    train_or_eval: str = "train"
    headless: bool = True

    # Rollout & Timesteps
    num_envs: int = 1024
    num_eval_envs: int = 1024
    total_timesteps: int = 1500
    learning_starts: int = 10
    num_steps: int = 1

    # Replay, Batching, Discounting
    buffer_size: int = 20480
    batch_size: int = 32768
    gamma: float = 0.99
    tau: float = 0.1

    # Update Schedule
    policy_frequency: int = 2
    num_updates: int = 12

    # Optimizer & Network
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    weight_decay: float = 0.1
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    init_scale: float = 0.01
    num_atoms: int = 101

    # Value Distribution & Exploration
    v_min: float = -250.0
    v_max: float = 250.0
    policy_noise: float = 0.001
    std_min: float = 0.001
    std_max: float = 0.4
    noise_clip: float = 0.5

    # Algorithm Flags
    use_cdq: bool = True
    compile: bool = True
    obs_normalization: bool = True
    max_grad_norm: float = 0.0
    amp: bool = True
    amp_dtype: str = "fp16"
    disable_bootstrap: bool = False
    measure_burnin: int = 3

    # Logging & Checkpointing
    wandb_project: str = "get_started_fttd3"
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    checkpoint_path: Optional[str] = None
    eval_interval: int = 700
    save_interval: int = 700
    video_width: int = 1024
    video_height: int = 1024
    max_iterations: int = 50000

    # Model directory
    model_dir: Optional[str] = None

    # Extra fields used by some task-specific configs
    state_file_path: Optional[str] = None

    def __post_init__(self) -> None:
        import os

        if self.model_dir is None:
            self.model_dir = os.path.join("outputs", self.exp_name, self.task)


__all__ = ["FastTD3Config"]
