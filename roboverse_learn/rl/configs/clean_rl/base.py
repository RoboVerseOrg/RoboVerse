from typing import Literal, Optional

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
class BaseRLConfig:
    """Base configuration for all RL algorithms in RoboVerse."""

    # Experiment
    exp_name: str = "rl_experiment"
    seed: int = 1
    torch_deterministic: bool = True

    # Device
    cuda: bool = True
    device: str = "cuda:0"

    # Task & Environment
    task: str = "walk_g1_dof29"
    robot: str = "g1_dof29"
    sim: SimBackend = "isaacgym"
    num_envs: int = 4096
    headless: bool = False

    # Training
    max_iterations: int = 50000
    save_interval: int = 100

    # Logging
    use_wandb: bool = False
    wandb_project: str = "roboverse_rl"
    wandb_entity: Optional[str] = None

    # Model directory
    model_dir: Optional[str] = None

    def __post_init__(self) -> None:
        import os

        if self.model_dir is None:
            self.model_dir = os.path.join("outputs", self.exp_name, self.task)
