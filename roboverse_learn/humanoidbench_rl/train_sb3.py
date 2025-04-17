from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable

import numpy as np
import rootutils
import wandb
import yaml
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def load_config_from_yaml(config_name: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_name (str): Name of the YAML config file

    Returns:
        dict: The loaded config dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["batch_size"] = config["num_envs"] * config["n_steps"] // config["num_batch"]
    return config


def get_lr_schedule(config: dict) -> float | Callable:
    """
    Create a learning rate schedule based on configuration.

    Args:
        config (dict): Configuration dictionary containing learning rate settings

    Returns:
        Union[float, Callable]: Constant learning rate or schedule function
    """
    # Get base learning rate
    base_lr = config.get("learning_rate", 0.0003)

    # Check if learning rate schedule is enabled
    if not config.get("use_lr_schedule", False):
        return base_lr

    # Get schedule type
    schedule_type = config.get("lr_schedule_type", "linear")

    # Get final learning rate as a fraction of initial
    final_lr_fraction = config.get("final_lr_fraction", 0.1)
    final_lr = base_lr * final_lr_fraction

    # For linear schedule
    if schedule_type == "linear":
        from stable_baselines3.common.utils import get_linear_fn

        return get_linear_fn(base_lr, final_lr, 1.0)

    # For cosine schedule
    elif schedule_type == "cosine":

        def func(progress_remaining: float) -> float:
            progress = 1.0 - progress_remaining
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            return final_lr + cosine_factor * (base_lr - final_lr)

        return func

    # For constant schedule (explicitly handled)
    elif schedule_type == "constant":
        log.info("Using constant learning rate schedule")
        return base_lr

    # Default to constant if unknown schedule type
    log.warning(f"Unknown learning rate schedule type: {schedule_type}. Using constant learning rate.")
    return base_lr


def main():
    if len(sys.argv) < 2:
        log.error("Please provide the config file path, e.g. python train_sb3.py configs/isaacgym.yaml")
        exit(1)
    config_name = sys.argv[1]
    config = load_config_from_yaml(config_name)
    log.info(f"Load config: {config_name}")

    if config.get("sim") == "isaacgym":
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    if config.get("use_wandb"):
        run = wandb.init(
            project=config.get("wandb_project", "humanoidbench_rl_training"),
            entity=config.get("wandb_entity"),
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
            name=f"SB3-{time.strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    else:
        from collections import namedtuple

        Run = namedtuple("Run", ["id"])
        run = Run(id=int(time.time()))

    # Create scenario config
    from metasim.cfg.scenario import ScenarioCfg

    scenario = ScenarioCfg(
        task=config.get("task"),
        robot=config.get("robot"),
        try_add_table=config.get("add_table", False),
        sim=config.get("sim"),
        num_envs=config.get("num_envs", 1),
        headless=True if config.get("train_or_eval") == "train" else False,
        cameras=[],
    )

    # For different simulators, the decimation factor is different, so we need to set it here
    scenario.task.decimation = config.get("decimation", 1)

    from roboverse_learn.humanoidbench_rl.wrapper_sb3 import Sb3EnvWrapper

    if config.get("sim") == "mujoco":
        if config.get("num_envs") > 1:
            log.error("Mujoco does not support multiple environments > 1")
            exit()
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "isaacgym":
        env = Sb3EnvWrapper(scenario=scenario)
    elif config.get("sim") == "isaaclab":
        env = Sb3EnvWrapper(scenario=scenario)
    else:
        raise ValueError(f"Invalid sim type: {config.get('sim')}")

    # Create learning rate schedule
    learning_rate = get_lr_schedule(config)
    if callable(learning_rate):
        log.info(f"Using {config.get('lr_schedule_type', 'linear')} learning rate schedule")
        log.info(f"Initial learning rate: {config.get('learning_rate', 0.0003)}")
        log.info(f"Final learning rate: {config.get('learning_rate', 0.0003) * config.get('final_lr_fraction', 0.1)}")
    else:
        log.info(f"Using constant learning rate: {learning_rate}")

    # Initialize PPO algorithm
    from stable_baselines3 import PPO

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        verbose=1,
        tensorboard_log=f"./ppo_logs/{run.id}",
        device="cpu",
    )

    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLogCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.returns_info = {
                "results/return": [],
                "results/episode_length": [],
                "results/success": [],
                "results/success_subtasks": [],
            }

        def _on_step(self) -> bool:
            infos = self.locals["infos"]
            for idx in range(len(infos)):
                curr_info = infos[idx]
                if "episode" in curr_info:
                    self.returns_info["results/return"].append(curr_info["episode"]["r"])
                    self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                    cur_info_success = curr_info.get("success", 0)
                    self.returns_info["results/success"].append(cur_info_success)
                    # cur_info_success_subtasks = curr_info.get("success_subtasks", 0)
                    # self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
            return True

        def _on_rollout_end(self) -> None:
            # Use the model's num_timesteps (cumulative across envs) as the global step.
            global_step = self.model.num_timesteps
            for key in self.returns_info.keys():
                if self.returns_info[key]:
                    self.logger.record(key, np.mean(self.returns_info[key]), global_step)
                    self.returns_info[key] = []

    class SaveModelCallback(BaseCallback):
        """
        Callback for saving the model every 1M timesteps.

        Args:
            save_path (str): Path to the directory where models will be saved
            save_freq (int): Frequency in timesteps at which to save the model
            verbose (int): Verbosity level
        """

        def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.save_path = save_path
            self.save_freq = save_freq
            self.last_save_step = 0

        def _init_callback(self) -> None:
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            # Check if it's time to save the model
            if self.num_timesteps - self.last_save_step >= self.save_freq:
                path = os.path.join(self.save_path, f"model_{self.num_timesteps}")
                self.model.save(path)
                log.info(f"Model saved to {path}")
                self.last_save_step = self.num_timesteps
            return True

    if config.get("train_or_eval") == "train":
        log.info("Starting training...")

        # Set up save directory
        save_dir = f"{config.get('model_save_path')}/{run.id}"
        os.makedirs(save_dir, exist_ok=True)

        model.learn(
            total_timesteps=config.get("total_timesteps", 1_000_000),
            log_interval=1,
            callback=[
                EpisodeLogCallback(),
                SaveModelCallback(save_path=save_dir, save_freq=config.get("model_save_freq", 1_000_000)),
            ],
            progress_bar=True,
        )
    elif config.get("train_or_eval") == "eval":
        # Load the trained model
        model.load(config.get("model_path"))

        # Evaluate the agent
        log.info("Starting evaluation...")
        obs = env.reset()
        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            env.render()
            if done:
                obs = env.reset()
                import matplotlib.pyplot as plt

                plt.plot(rewards)
                plt.savefig("rewards.png")
                plt.clf()
                rewards = []

    # Close environment and wandb
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
