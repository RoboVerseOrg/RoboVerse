from __future__ import annotations

from typing import Dict, List, Literal, Optional

from metasim.utils import configclass

SimBackend = Literal[
    "isaacgym",
    "isaacsim",
    "isaaclab",
    "mujoco",
    "genesis",
    "mjx",
]

from roboverse_learn.rl.configs.rsl_rl.algorithm import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPPOConfig(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration mirroring Unitree on-policy train config."""

    # Experiment / runner settings
    exp_name: str = "rsl_rl_ppo"
    experiment_name: str = ""  # defaults to task name if left empty
    run_name: str = ""
    seed: int = 1
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    empirical_normalization: bool = False
    obs_groups: Optional[Dict[str, List[str]]] = None
    clip_actions: Optional[float] = None
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "rsl_rl_ppo"
    resume: Optional[str] = None
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    checkpoint: int = -1

    # Environment / device
    task: str = "walk_g1_dof29"
    robot: str = "g1_dof29"
    sim: SimBackend = "isaacgym"
    num_envs: int = 4096
    headless: bool = False
    cuda: bool = True
    device: str = "cuda:0"
    torch_deterministic: bool = True

    # Logging
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    model_dir: Optional[str] = None
    train_cfg: Optional[dict] = None

    # Policy configuration
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Algorithm configuration
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self) -> None:
        import os

        if not self.experiment_name:
            self.experiment_name = self.task

        if self.model_dir is None:
            name = self.exp_name or self.experiment_name
            self.model_dir = os.path.join("outputs", name, self.task)

        if self.obs_groups is None:
            self.obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}

        # Build runner training config for RSL-RL
        policy_cfg = self.policy.to_dict() if hasattr(self.policy, "to_dict") else dict(self.policy.__dict__)
        algo_cfg = self.algorithm.to_dict() if hasattr(self.algorithm, "to_dict") else dict(self.algorithm.__dict__)

        self.train_cfg = {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "empirical_normalization": self.empirical_normalization,
            "run_name": self.run_name,
            "logger": self.logger,
            "neptune_project": self.neptune_project,
            "wandb_project": self.wandb_project,
            "resume": bool(self.resume),
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            "obs_groups": self.obs_groups,
            "policy": policy_cfg,
            "algorithm": algo_cfg,
        }

        if self.clip_actions is not None:
            self.train_cfg["clip_actions"] = self.clip_actions


__all__ = ["RslRlPPOConfig"]
