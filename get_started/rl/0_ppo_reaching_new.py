"""Train PPO for reaching task."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import rootutils
import torch
import tyro
from gymnasium import spaces
from loguru import logger as log
from packaging.version import Version
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.utils import ObsSaver
from metasim.utils.setup_util import register_task
from metasim.wrapper.gym_vec_env import MetaSimVecEnv


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 16
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "isaaclab"


args = tyro.cli(Args)


class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robots[0].joint_limits

        # TODO: customize action space?
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            dtype=np.float32,
        )

        # TODO: customize observation space?
        # Observation space: joint positions + end effector position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits) + 3,),  # joints + XYZ
            dtype=np.float32,
        )

        self.env = env
        self.render_mode = None  # XXX
        super().__init__(self.env.num_envs, self.observation_space, self.action_space)

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        self.action_dicts = [
            {
                self.env.scenario.robots[0].name: {
                    "dof_pos_target": dict(zip(self.env.scenario.robots[0].joint_limits.keys(), action))
                }
            }
            for action in actions
        ]

    def step_wait(self):
        """Wait for the step to complete."""
        obs, rewards, success, timeout, _ = self.env.step(self.action_dicts)

        dones = success | timeout.to(success.device)
        if dones.any():
            self.env.reset(env_ids=dones.nonzero().squeeze(-1).tolist())

        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if dones[env_id]:
                extra[env_id]["terminal_observation"] = obs[env_id].cpu().numpy()
            extra[env_id]["TimeLimit.truncated"] = timeout[env_id].item() and not success[env_id].item()

        obs = self.env.unwrapped._get_obs()

        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), extra

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

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
        return [getattr(self.env.handler, attr_name)] * len(indices)

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
    if Version(gym.__version__) < Version("1"):
        metasim_env = gym.make(args.task, num_envs=args.num_envs, sim=args.sim)
    else:
        metasim_env = gym.make_vec(args.task, num_envs=args.num_envs, sim=args.sim)
    scenario = metasim_env.scenario
    env = StableBaseline3VecEnv(metasim_env)
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
    # add cameras to the scenario
    args.num_envs = 16
    metasim_env = gym.make(args.task, num_envs=args.num_envs, sim=args.sim)
    task_name = scenario.task.__class__.__name__[:-3]
    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}.mp4")
    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{task_name}_{args.sim}")

    # inference
    obs, _ = metasim_env.reset()
    obs_orin = metasim_env.env.handler.get_states()
    obs_saver.add(obs_orin)
    for _ in range(100):
        actions, _ = model.predict(obs.cpu().numpy(), deterministic=True)
        action_dicts = [
            {"dof_pos_target": dict(zip(metasim_env.scenario.robots[0].joint_limits.keys(), action))}
            for action in actions
        ]
        obs, _, _, _, _ = metasim_env.step(action_dicts)

        obs_orin = metasim_env.env.handler.get_states()
        obs_saver.add(obs_orin)
    obs_saver.save()


if __name__ == "__main__":
    train_ppo()
