"""Train PPO for reaching task."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rootutils
import torch
import tyro
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.utils import ObsSaver
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class


@dataclass
class Args(ScenarioCfg):
    """Arguments for training PPO."""

    task: str = "debug:reach_far_away"
    robot: str = "franka"
    num_envs: int = 16
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "isaaclab"


args = tyro.cli(Args)


class MetaSimVecEnv(VectorEnv):
    """Vectorized environment for MetaSim that supports parallel RL training."""

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        sim: str = "isaaclab",
        task_name: str | None = None,
        num_envs: int | None = 4,
    ):
        """Initialize the environment."""
        if scenario is None:
            scenario = ScenarioCfg(task="pick_cube", robots=["franka"])
            scenario.task = task_name
            scenario.num_envs = num_envs
            scenario = ScenarioCfg(**vars(scenario))
        self.num_envs = scenario.num_envs
        env_class = get_sim_env_class(SimType(sim))
        env = env_class(scenario)
        self.env: EnvWrapper[BaseSimHandler] = env
        self.render_mode = None  # XXX
        self.scenario = scenario

        # Get candidate states
        self.candidate_init_states, _, _ = get_traj(scenario.task, scenario.robots[0])

        # XXX: is the inf space ok?
        self.single_observation_space = spaces.Box(-np.inf, np.inf)
        self.single_action_space = spaces.Box(-np.inf, np.inf)

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self, env_ids: list[int] | None = None, seed: int | None = None):
        """Reset the environment."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        init_states = self.unwrapped._get_default_states(seed)
        self.env.reset(states=init_states, env_ids=env_ids)
        return self.unwrapped._get_obs(), {}

    def step(self, actions: list[dict]):
        """Step the environment."""
        _, _, success, timeout, _ = self.env.step(actions)
        obs = self.unwrapped._get_obs()
        rewards = self.unwrapped._calculate_rewards()
        return obs, rewards, success, timeout, {}

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Helper methods
    ############################################################
    def _get_obs(self):
        ## TODO: put this function into task definition?
        ## TODO: use torch instead of numpy
        """Get current observations for all environments."""
        states = self.env.handler.get_states()
        joint_pos = states.robots["franka"].joint_pos
        panda_hand_index = states.robots["franka"].body_names.index("panda_hand")
        ee_pos = states.robots["franka"].body_state[:, panda_hand_index, :3]

        return torch.cat([joint_pos, ee_pos], dim=1)

    def _calculate_rewards(self):
        """Calculate rewards based on distance to origin."""
        states = self.env.handler.get_states()
        tot_reward = torch.zeros(self.num_envs, device=self.env.handler.device)
        for reward_fn, weight in zip(self.scenario.task.reward_functions, self.scenario.task.reward_weights):
            tot_reward += weight * reward_fn(states, self.scenario.robots[0].name)
        return tot_reward

    def _get_default_states(self, seed: int | None = None):
        """Generate default reset states."""
        ## TODO: use non-reqeatable random choice when there is enough candidate states?
        return random.Random(seed).choices(self.candidate_init_states, k=self.num_envs)


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
    ## Choice 1: use scenario config to initialize the environment
    scenario = ScenarioCfg(task=args.task, robots=[args.robot], sim=args.sim, num_envs=args.num_envs)
    scenario.cameras = []  # XXX: remove cameras to avoid rendering to speed up
    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)

    ## Choice 2: use gym.make to initialize the environment
    # metasim_env = gym.make("reach_origin", num_envs=args.num_envs)
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
    scenario = ScenarioCfg(task=args.task, robots=[args.robot], sim=args.sim, num_envs=args.num_envs)
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    metasim_env = MetaSimVecEnv(scenario, task_name=args.task, num_envs=args.num_envs, sim=args.sim)
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
