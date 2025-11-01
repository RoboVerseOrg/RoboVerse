"""Train PPO for reaching task using RLTaskEnv."""

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
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

# Ensure reaching tasks are registered exactly once from the canonical module
# from metasim.task.gym_registration import make_vec
from gymnasium import make_vec

import metasim  # noqa: F401
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.obs_utils import ObsSaver


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 128
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "mjx"
    headless: bool = False
    device: str = "cuda"
    enable_viser: bool = False  # Enable real-time 3D visualization with Viser


args = tyro.cli(Args)


def _to_np(x):
    """Convert torch tensors to CPU numpy, else asarray."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class VecEnvWrapper(VecEnv):
    """Bridge Gymnasium VectorEnv to SB3 VecEnv."""

    def __init__(self, gym_vec_env):
        """Initialize the wrapper."""
        self.gym_vec = gym_vec_env
        super().__init__(
            num_envs=gym_vec_env.num_envs,
            observation_space=gym_vec_env.observation_space,
            action_space=gym_vec_env.action_space,
        )
        self._last_obs = None

    def reset(self):
        """Reset all envs and return observations."""
        obs, _ = self.gym_vec.reset()
        obs = _to_np(obs)
        self._last_obs = obs.copy()
        return obs

    def step_async(self, actions):
        """Send actions to the vectorized env."""
        self.actions = torch.tensor(actions, device=self.gym_vec.device, dtype=torch.float32)

    def step_wait(self):
        """Step the envs and adapt outputs for SB3."""
        obs, rewards, terminated, truncated, infos = self.gym_vec.step(self.actions)

        obs = _to_np(obs)
        rewards = _to_np(rewards).reshape(self.num_envs)
        terminated = _to_np(terminated).astype(bool).reshape(self.num_envs)
        truncated = _to_np(truncated).astype(bool).reshape(self.num_envs)
        dones = np.logical_or(terminated, truncated)

        if isinstance(infos, list):
            out_infos = []
            n = len(infos) if infos is not None else 0
            for i in range(self.num_envs):
                info_i = dict(infos[i]) if (infos is not None and i < n and infos[i] is not None) else {}
                info_i["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
                if dones[i] and self._last_obs is not None:
                    info_i["terminal_observation"] = np.array(self._last_obs[i], copy=True)
                out_infos.append(info_i)
        else:
            base = dict(infos) if infos is not None else {}
            out_infos = []
            for i in range(self.num_envs):
                info_i = dict(base)
                info_i["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
                if dones[i] and self._last_obs is not None:
                    info_i["terminal_observation"] = np.array(self._last_obs[i], copy=True)
                out_infos.append(info_i)

        self._last_obs = obs.copy()
        return obs, rewards, dones, out_infos

    def close(self):
        """Close the underlying envs."""
        self.gym_vec.close()

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        """Not implemented."""
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Not implemented."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Not implemented."""
        raise NotImplementedError


def train_ppo():
    """Train PPO for reaching task using RLTaskEnv."""

    # Create scenario configuration
    env_id = f"RoboVerse/{args.task}"
    env = make_vec(
        env_id,
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[],
        device=args.device,
    )

    # Optionally wrap with Viser visualization
    if args.enable_viser:
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        env = TaskViserWrapper(env)

    # Create VecEnv wrapper for SB3
    env = VecEnvWrapper(env)

    log.info(f"Created environment with {env.num_envs} environments")

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
    model.learn(total_timesteps=1_00_0000)

    # Save the model

    model.save(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    env.close()

    # Inference and Save Video
    # Create new environment for inference
    env_inference = make_vec(
        env_id,
        robots=[args.robot],
        simulator=args.sim,
        num_envs=1,  # Use single environment for inference
        headless=True,  # Always headless for inference to save video
        cameras=[PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))],
        device=args.device,
    )

    # Optionally wrap inference environment with Viser visualization
    if args.enable_viser:
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        env_inference = TaskViserWrapper(env_inference)

    env_inference = VecEnvWrapper(env_inference)

    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{args.sim}.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    # inference
    obs = env_inference.reset()
    obs_orin = env_inference.gym_vec.task_env.handler.get_states()
    obs_saver.add(obs_orin)

    for _ in range(250):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()

        obs_orin = env_inference.gym_vec.task_env.handler.get_states()
        obs_saver.add(obs_orin)

    obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
