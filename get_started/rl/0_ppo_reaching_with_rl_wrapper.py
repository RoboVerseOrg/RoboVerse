"""Train PPO for reaching task using RLTaskWrapper."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rootutils
import torch
import tyro
from gymnasium import spaces
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.rl.fast_td3.task_wrapper import RLTaskWrapper
from get_started.utils import ObsSaver
from metasim.utils.setup_util import register_task
from scenario_cfg.scenario import ScenarioCfg


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 16
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "isaaclab"


args = tyro.cli(Args)


class RLTaskWrapperVecEnv(VecEnv):
    """Vectorized environment wrapper for RLTaskWrapper to work with Stable Baselines 3."""

    def __init__(self, rl_wrapper: RLTaskWrapper):
        """Initialize the environment."""
        self.rl_wrapper = rl_wrapper

        # Set up action space based on RLTaskWrapper
        action_space_info = rl_wrapper.action_space
        self.action_space = spaces.Box(
            low=action_space_info["low"],
            high=action_space_info["high"],
            shape=action_space_info["shape"],
            dtype=action_space_info["dtype"],
        )

        # Set up observation space based on RLTaskWrapper
        obs_space_info = rl_wrapper.observation_space
        self.observation_space = spaces.Box(
            low=obs_space_info["low"],
            high=obs_space_info["high"],
            shape=obs_space_info["shape"],
            dtype=obs_space_info["dtype"],
        )

        super().__init__(rl_wrapper.num_envs, self.observation_space, self.action_space)
        self.render_mode = None

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs = self.rl_wrapper.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Convert numpy actions to torch
        self.pending_actions = torch.tensor(actions, device=self.rl_wrapper.device, dtype=torch.float32)

    def step_wait(self):
        """Wait for the step to complete."""
        obs, reward, done, info = self.rl_wrapper.step(self.pending_actions)

        # Convert to numpy for SB3
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()
        done_np = done.cpu().numpy()

        # Prepare extra info for SB3
        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if done[env_id]:
                extra[env_id]["terminal_observation"] = obs_np[env_id]
            extra[env_id]["TimeLimit.truncated"] = info["time_outs"][env_id].item() and not done[env_id].item()

        return obs_np, reward_np, done_np, extra

    def render(self):
        """Render the environment."""
        return self.rl_wrapper.render()

    def close(self):
        """Close the environment."""
        self.rl_wrapper.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        """Get images from the environment."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.rl_wrapper.env, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set an attribute of the environment."""
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call a method of the environment."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if the environment is wrapped by a given wrapper class."""
        raise NotImplementedError


def train_ppo():
    """Train PPO for reaching task using RLTaskWrapper."""
    register_task(args.task)

    # Create scenario configuration
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        sim=args.sim,
        num_envs=args.num_envs,
        headless=True,
        cameras=[],
    )

    # Create RLTaskWrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_wrapper = RLTaskWrapper(scenario, device=device)

    # Set appropriate episode length
    rl_wrapper.max_episode_steps = 500  # Adjust based on your task

    # Create VecEnv wrapper for SB3
    env = RLTaskWrapperVecEnv(rl_wrapper)

    log(f"Created environment with {env.num_envs} environments")
    log(f"Observation space: {env.observation_space}")
    log(f"Action space: {env.action_space}")
    log(f"Max episode steps: {rl_wrapper.max_episode_steps}")

    # PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    model.learn(total_timesteps=1_000_000)

    # Save the model
    task_name = scenario.task.__class__.__name__[:-3]
    model.save(f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}_rl_wrapper")

    env.close()

    # Inference and Save Video
    # Create new environment for inference
    scenario_inference = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        sim=args.sim,
        num_envs=16,
        headless=True,
        cameras=[],
    )

    rl_wrapper_inference = RLTaskWrapper(scenario_inference, device=device)
    rl_wrapper_inference.max_episode_steps = 500
    env_inference = RLTaskWrapperVecEnv(rl_wrapper_inference)

    task_name = scenario.task.__class__.__name__[:-3]
    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}_rl_wrapper.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}_rl_wrapper")

    # inference
    obs = env_inference.reset()
    obs_orin = rl_wrapper_inference.env.get_states()
    obs_saver.add(obs_orin)

    for _ in range(100):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()

        obs_orin = rl_wrapper_inference.env.get_states()
        obs_saver.add(obs_orin)

    obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
