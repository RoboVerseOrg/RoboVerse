import os
import wandb
import pathlib
from typing import Literal, Optional
from loguru import logger as log

from metasim.utils import configclass
from roboverse_learn.rl.configs.rsl_rl.ppo import RslRlPPOConfig
from roboverse_learn.rl.configs.rsl_rl.algorithm import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


SimBackend = Literal[
    "isaacgym",
    "isaacsim",
    "mujoco",
    "genesis",
    "mjx",
    "newton",
]


@configclass
class RslRlPPOTrackingConfig(RslRlPPOConfig):
    """RSL-RL PPO configs for motion tracking task."""
    # Experiment / runner settings
    exp_name: str = "rsl_rl_ppo_tracking"
    max_iterations = 30000
    save_interval = 500
    empirical_normalization = True  # deprecated
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    wandb_project: str = "rsl_rl_ppo_tracking"

    # Environment / device
    # Use RoboVerse/MetaSim task by default (runs via metasim's IsaacSim handler).
    task = "motion-tracking"
    robot = "g1_tracking"
    sim: SimBackend = "isaacsim"

    # Logging
    use_wandb: bool = True

    # WandB registry for loading motions (training)
    registry_name: Optional[str] = None

    # WandB run path for loading model and motions (evaluation)
    wandb_path: Optional[str] = None

    # Motion file path (gets overridden except when loading from local file)
    motion_file: Optional[str] = None

    # Policy configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # entropy_coef=0.05,  # FIXME was a typo; high `entropy_coef` leads to high entropy loss, low converged mean reward, and short episode length
        entropy_coef=0.005,  # NOTE the only difference
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
        """`motion_file` will point to an existing motion file path after `__post_init__()`."""
        super().__post_init__()

        if self.registry_name:
            if ":" not in self.registry_name:  # Check if the registry name includes alias, if not, append ":latest"
                self.registry_name += ":latest"

            api = wandb.Api()
            artifact = api.artifact(self.registry_name)
            self.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

        elif self.wandb_path:
            if "model" in self.wandb_path:
                run_path = "/".join(self.wandb_path.split("/")[:-1])
                # e.g., "org/project/run_id/model_1000.pt" yields "org/project/run_id"
            else:
                run_path = self.wandb_path

            wandb_run = wandb.Api().run(run_path)
            if "model" in self.wandb_path:
                # use specified model file
                model_file = self.wandb_path.split("/")[-1]
            else:
                # files are formatted as model_xxx.pt, find the largest filename (max iter)
                files = [file.name for file in wandb_run.files() if "model" in file.name]
                model_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

            # prepare log dir to store temporary files
            log_dir = f"./outputs/{self.robot}/{self.task}/temp"
            os.makedirs(log_dir, exist_ok=True)

            wandb_file = wandb_run.file(str(model_file))
            wandb_file.download(log_dir, replace=True)
            log.info(f"Loading checkpoint from {run_path}/{model_file}")
            self.checkpoint_path = f"{log_dir}/{model_file}"

            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            assert art, "No motion artifact found in WandB run"
            self.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

        else:
            assert self.motion_file, "Provide local motion file path if not loading from WandB"
