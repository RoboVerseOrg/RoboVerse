from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces
from loguru import logger as log
from packaging.version import Version
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from metasim.utils.setup_util import register_task
from metasim.wrapper.gym_vec_env import MetaSimVecEnv

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@dataclass
class Args:
    task: str = "debug:reach_origin"
    robot: str = "franka"
    num_envs: int = 16
    sim: Literal["isaaclab", "isaacgym", "mujoco"] = "isaaclab"


args = tyro.cli(Args)


class StableBaseline3VecEnv(VecEnv):
    def __init__(self, env: MetaSimVecEnv):
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
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self.action_dicts = [
            {args.robot: {"dof_pos_target": dict(zip(self.env.scenario.robots[0].joint_limits.keys(), action))}}
            for action in actions
        ]

    def step_wait(self):
        obs, rewards, success, timeout, _ = self.env.step(self.action_dicts)

        dones = success | timeout
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
        return self.env.render()

    def close(self):
        self.env.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError


def train_ppo():
    ## Choice 1: use scenario config to initialize the environment
    # scenario = ScenarioCfg(task=args.task, robots=[args.robot], num_envs=args.num_envs, sim=args.sim)
    # scenario.cameras = []  # XXX: remove cameras to avoid rendering to speed up
    # metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)

    ## Choice 2: use gym.make to initialize the environment
    register_task(args.task)
    if Version(gym.__version__) < Version("1"):
        metasim_env = gym.make(args.task, num_envs=args.num_envs, sim=args.sim)
    else:
        metasim_env = gym.make_vec(args.task, num_envs=args.num_envs, sim=args.sim)
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
    model.save("ppo_reach")


if __name__ == "__main__":
    train_ppo()
