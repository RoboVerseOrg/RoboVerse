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
from get_started.utils import ObsSaver
from scenario_cfg.cameras import PinholeCameraCfg
from scenario_cfg.scenario import ScenarioCfg
from tasks.registry import load_task
from tasks.rl_task import RLTaskEnv


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 128
    sim: Literal["isaacsim", "isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "mjx"
    headless: bool = False


args = tyro.cli(Args)


class VecEnvWrapper(VecEnv):
    """Vectorized environment wrapper for RLTaskEnv to work with Stable Baselines 3."""

    def __init__(self, rl_env: RLTaskEnv):
        """Initialize the environment."""
        self.rl_env = rl_env

        # Use action space directly from RLTaskEnv
        self.action_space = rl_env.action_space

        # Use observation space directly from RLTaskEnv
        self.observation_space = rl_env.observation_space

        super().__init__(rl_env.num_envs, self.observation_space, self.action_space)
        self.render_mode = None

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs, _ = self.rl_env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Convert numpy actions to torch
        self.pending_actions = torch.tensor(actions, device=self.rl_env.device, dtype=torch.float32)

    def step_wait(self):
        """Wait for the step to complete."""
        obs, reward, terminated, time_out, info = self.rl_env.step(self.pending_actions)

        done = terminated | time_out
        # Convert to numpy for SB3
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()
        done_np = done.cpu().numpy()

        # Prepare extra info for SB3
        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if bool(done[env_id].item()):
                extra[env_id]["terminal_observation"] = obs_np[env_id]
            extra[env_id]["TimeLimit.truncated"] = bool(time_out[env_id].item() and not terminated[env_id].item())

        return obs_np, reward_np, done_np, extra

    def render(self):
        """Render the environment."""
        return self.rl_env.render()

    def close(self):
        """Close the environment."""
        self.rl_env.close()

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
    """Train PPO for reaching task using RLTaskEnv."""

    # Create scenario configuration
    # scenario = ScenarioCfg(
    #     robots=[args.robot],
    #     simulator=args.sim,
    #     num_envs=args.num_envs,
    #     headless=args.headless,
    #     cameras=[],
    # )

    # # Create RLTaskEnv via registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # rl_env = load_task(args.task, scenario, device=device)

    # # Create VecEnv wrapper for SB3
    # env = VecEnvWrapper(rl_env)

    # log.info(f"Created environment with {env.num_envs} environments")
    # log.info(f"Observation space: {env.observation_space}")
    # log.info(f"Action space: {env.action_space}")
    # log.info(f"Max episode steps: {rl_env.max_episode_steps}")

    # PPO configuration
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     n_steps=128,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # # Start training
    # model.learn(total_timesteps=2_000_000)

    # # Save the model

    # model.save(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    # env.close()

    # Inference and Save Video
    # Create new environment for inference
    scenario_inference = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=1,
        # headless=True,
        headless=False,
        cameras=[PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))],
    )

    rl_env_inference = load_task(args.task, scenario_inference, device=device)
    env_inference = VecEnvWrapper(rl_env_inference)

    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{args.sim}.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    # inference
    obs = env_inference.reset()
    obs_orin = rl_env_inference.env.get_states()
    obs_saver.add(obs_orin)

    for _ in range(200):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()

        obs_orin = rl_env_inference.env.get_states()
        obs_saver.add(obs_orin)

    obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
