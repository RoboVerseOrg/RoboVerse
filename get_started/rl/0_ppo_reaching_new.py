"""Train PPO for reaching task."""

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

from get_started.utils import ObsSaver
from scenario_cfg.scenario import ScenarioCfg
from metasim.utils.setup_util import register_task
from roboverse_learn.tasks.base import BaseTaskWrapper


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 16
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "isaaclab"


args = tyro.cli(Args)


class StableBaseline3VecEnv(BaseTaskWrapper, VecEnv):
    """Vectorized environment for Stable Baselines 3 that inherits from BaseTaskWrapper."""

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None):
        """Initialize the environment."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize BaseTaskWrapper
        super().__init__(scenario)

        # Initialize VecEnv
        joint_limits = scenario.robots[0].joint_limits

        # Set up action space
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            dtype=np.float32,
        )

        # Set up observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits) + 3,),  # joints + XYZ
            dtype=np.float32,
        )

        # Initialize episode step counter for timeout
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_steps = 1000  # Set appropriate episode length

        # Initialize VecEnv
        super(VecEnv, self).__init__(self.num_envs, self.observation_space, self.action_space)

        self.render_mode = None

    def _observation_space(self) -> dict:
        """Get the observation space of the environment."""
        return {
            "shape": (self.observation_space.shape[0],),
            "low": self.observation_space.low,
            "high": self.observation_space.high,
            "dtype": np.float32,
        }

    def _action_space(self) -> dict:
        """Get the action space of the environment."""
        return {
            "shape": (self.action_space.shape[0],),
            "low": self.action_space.low,
            "high": self.action_space.high,
            "dtype": np.float32,
        }

    def _observation(self, env_states) -> torch.Tensor:
        """Get the observation from environment states."""
        # This should be implemented based on your task's observation function
        # For now, return a placeholder
        if hasattr(self.env.task, "get_obs"):
            return self.env.task.get_obs(env_states).to(self.device)
        else:
            # Fallback: return flattened states
            return torch.tensor(env_states, device=self.device).flatten(start_dim=1)

    def _privileged_observation(self, env_states) -> torch.Tensor:
        """Get the privileged observation of the environment."""
        return self._observation(env_states)

    def _reward(self, env_states) -> torch.Tensor:
        """Get the reward of the environment."""
        if hasattr(self.env.task, "get_reward"):
            return self.env.task.get_reward(env_states).to(self.device)
        else:
            # Fallback: return zero reward
            return torch.zeros(self.num_envs, device=self.device)

    def _terminated(self, env_states) -> torch.Tensor:
        """Get the terminated flag of the environment."""
        if hasattr(self.env.task, "get_terminated"):
            return self.env.task.get_terminated(env_states).to(self.device)
        else:
            # Fallback: return False
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _time_out(self, env_states) -> torch.Tensor:
        """Get the timeout flag of the environment based on max episode length."""
        timeout_flag = self._episode_steps >= self.max_episode_steps
        return timeout_flag

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        # Reset episode step counter
        self._episode_steps.zero_()

        # Use BaseTaskWrapper reset
        obs, _, _ = super().reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Convert numpy actions to torch
        self.pending_actions = torch.tensor(actions, device=self.device, dtype=torch.float32)

    def step_wait(self):
        """Wait for the step to complete."""
        # Increment episode step counter
        self._episode_steps += 1

        # Use BaseTaskWrapper step
        obs, priv_obs, reward, terminated, time_out, _ = super().step(self.pending_actions)

        # Convert to numpy for SB3
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()

        # Combine termination and timeout
        dones = terminated | time_out
        dones_np = dones.cpu().numpy()

        # Reset completed episodes
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
            self.reset(env_ids=done_indices)

        # Prepare extra info for SB3
        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if dones[env_id]:
                extra[env_id]["terminal_observation"] = obs_np[env_id]
            extra[env_id]["TimeLimit.truncated"] = time_out[env_id].item() and not terminated[env_id].item()

        return obs_np, reward_np, dones_np, extra

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        super().close()

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
        return [getattr(self.env, attr_name)] * len(indices)

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
    """Train PPO for reaching task."""
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

    # Create environment using our new wrapper
    env = StableBaseline3VecEnv(scenario)

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
    model.save(f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}")

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

    env_inference = StableBaseline3VecEnv(scenario_inference)
    task_name = scenario.task.__class__.__name__[:-3]
    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}")

    # inference
    obs = env_inference.reset()
    obs_orin = env_inference.env.get_states()
    obs_saver.add(obs_orin)

    for _ in range(100):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()

        obs_orin = env_inference.env.get_states()
        obs_saver.add(obs_orin)

    obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
