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
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.task.rl_task import RLTaskEnv
from metasim.utils.obs_utils import ObsSaver


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 128
    sim: Literal["isaacsim", "isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "mjx"
    headless: bool = False
    enable_viser: bool = False  # Enable real-time 3D visualization with Viser


args = tyro.cli(Args)


class VecEnvWrapper(VecEnv):
    """Vectorized environment wrapper for RLTaskEnv to work with Stable Baselines 3."""

    def __init__(self, env: RLTaskEnv):
        """Initialize the environment."""
        self.env = env

        # Use action space directly from RLTaskEnv
        self.action_space = env.action_space

        # Use observation space directly from RLTaskEnv
        self.observation_space = env.observation_space

        super().__init__(env.num_envs, self.observation_space, self.action_space)
        self.render_mode = None

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Convert numpy actions to torch
        self.pending_actions = torch.tensor(actions, device=self.env.device, dtype=torch.float32)

    def step_wait(self):
        """Wait for the step to complete."""
        obs, reward, terminated, time_out, info = self.env.step(self.pending_actions)

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
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

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
    task_cls = get_task_class(args.task)
    # Get default scenario from task class and update with specific parameters
    scenario = task_cls.scenario.update(
        robots=[args.robot], simulator=args.sim, num_envs=args.num_envs, headless=args.headless, cameras=[]
    )

    # # Create RLTaskEnv via registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario=scenario)

    # Optionally wrap with Viser visualization
    if args.enable_viser:
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        env = TaskViserWrapper(env)

    # # Create VecEnv wrapper for SB3
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
    model.learn(total_timesteps=1_000_000)

    # Save the model

    model.save(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    env.close()

    # Inference and Save Video
    # Create new environment for inference using task's default scenario
    scenario_inference = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=1,
        headless=True,
        cameras=[PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))],
    )

    env_inference = task_cls(scenario_inference, device=device)

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
    obs_orin = env_inference.env.handler.get_states()
    obs_saver.add(obs_orin)

    for _ in range(250):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()
        obs_orin = env_inference.env.handler.get_states()
        obs_saver.add(obs_orin)

    # obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
