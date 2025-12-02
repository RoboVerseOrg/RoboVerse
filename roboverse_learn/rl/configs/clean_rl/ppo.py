from typing import Optional

from metasim.utils import configclass

from roboverse_learn.rl.configs.clean_rl.base import BaseRLConfig, SimBackend


@configclass
class CleanRLPPOConfig(BaseRLConfig):
    """CleanRL PPO configuration adapted for RoboVerse."""

    # Experiment
    exp_name: str = "clean_rl_ppo"

    # Tracking / logging flags (CleanRL-style)
    track: bool = False
    wandb_project: str = "cleanRL"
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Task & Environment overrides
    task: str = "stand"
    robot: str = "h1"
    sim: SimBackend = "mjx"
    headless: bool = False
    device: str = "cuda"

    # Algorithm-specific arguments (CleanRL naming)
    total_timesteps: int = 10000000
    learning_rate: float = 3e-4
    num_envs: int = 2048
    num_steps: int = 64
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 128
    update_epochs: int = 5
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Runtime-computed (filled in main script)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
