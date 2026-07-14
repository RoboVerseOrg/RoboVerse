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
    sim: Literal["isaacsim", "isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "isaacsim"
    headless: bool = True
    enable_viser: bool = True  # Enable real-time 3D visualization with Viser
    enable_rerun: bool = True  # Enable real-time 3D visualization with Rerun


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

        # Terminal-observation capture.
        #
        # ``RLTaskEnv.step`` auto-resets the done envs *in place* before returning, so the
        # observation it hands back for a done env is already the first observation of the
        # *next* episode, not the terminal one. SB3 corrects truncated episodes with
        # ``reward += gamma * V(terminal_observation)``, so reporting the post-reset
        # observation silently corrupts the final reward of every timed-out episode. The task
        # publishes no usable pre-reset observation (``info["observations"]["raw"]["obs"]``
        # holds the *first* obs of the current episode, not the terminal one), so we snapshot
        # it here: record every observation the task computes and keep the last one produced
        # before the auto-reset fires.
        self._task = env
        while not isinstance(self._task, RLTaskEnv) and hasattr(self._task, "env"):
            self._task = self._task.env  # unwrap viz/debug wrappers around the task
        self._latest_obs: torch.Tensor | None = None
        self._terminal_obs: torch.Tensor | None = None
        self._task_observation = self._task._observation
        self._task_reset = self._task.reset
        self._task._observation = self._capture_observation
        self._task.reset = self._capture_reset

    def _capture_observation(self, states):
        """Record the observation the task just computed (see ``__init__``)."""
        obs = self._task_observation(states)
        self._latest_obs = obs
        return obs

    def _capture_reset(self, *args, **kwargs):
        """Snapshot the pre-reset observation before a reset overwrites the simulator state."""
        if self._terminal_obs is None and self._latest_obs is not None:
            # ``step`` overwrites the obs tensor in place for the done envs right after this
            # reset returns, so keep a copy rather than a reference.
            self._terminal_obs = self._latest_obs.clone()
        return self._task_reset(*args, **kwargs)

    ############################################################
    ## Gym-like interface
    ############################################################
    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        self._terminal_obs = None
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""
        # Convert numpy actions to torch
        self.pending_actions = torch.tensor(actions, device=self.env.device, dtype=torch.float32)

    def step_wait(self):
        """Wait for the step to complete."""
        self._terminal_obs = None  # armed for the auto-reset that happens inside this step
        obs, reward, terminated, time_out, info = self.env.step(self.pending_actions)

        done = terminated | time_out
        # Convert to numpy for SB3
        obs_np = obs.cpu().numpy()
        reward_np = reward.cpu().numpy()
        done_np = done.cpu().numpy()

        terminal_obs_np = None
        if bool(done.any()):
            terminal_obs = self._terminal_obs
            if terminal_obs is None:
                # Tasks that override ``step`` may compute their observations without going
                # through ``_observation``; those are expected to publish the pre-reset
                # observation themselves (e.g. ``LeggedRobotTask``).
                terminal_obs = info.get("observations", {}).get("raw", {}).get("obs")
            if terminal_obs is None:
                raise RuntimeError(
                    "VecEnvWrapper: an env is done but no pre-reset observation was captured. SB3"
                    " bootstraps truncated episodes from `terminal_observation`, so handing it the"
                    " post-reset observation would silently corrupt the last reward of every"
                    " episode. The task must either compute observations through `_observation()`"
                    " or publish the pre-reset observation in info['observations']['raw']['obs']."
                )
            terminal_obs_np = terminal_obs.cpu().numpy()

        # Prepare extra info for SB3
        extra = [{} for _ in range(self.num_envs)]
        for env_id in range(self.num_envs):
            if bool(done[env_id].item()):
                extra[env_id]["terminal_observation"] = terminal_obs_np[env_id]
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

    # Optionally wrap with visualization
    if args.enable_viser or args.enable_rerun:
        from metasim.utils.viz_task_wrapper import TaskVizWrapper

        env = TaskVizWrapper(
            env,
            use_rerun=args.enable_rerun,
            use_viser=args.enable_viser,
            rerun_app_name="PPO Training",
            viser_port=8080,
            update_freq=10,
        )

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

    # Optionally wrap inference environment with visualization
    if args.enable_viser or args.enable_rerun:
        from metasim.utils.viz_task_wrapper import TaskVizWrapper

        env_inference = TaskVizWrapper(
            env_inference,
            use_rerun=args.enable_rerun,
            use_viser=args.enable_viser,
            rerun_app_name="PPO Inference",
            viser_port=8080,
            update_freq=1,
        )

    env_inference = VecEnvWrapper(env_inference)

    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{args.sim}.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    # inference
    obs = env_inference.reset()
    obs_orin = env_inference.env.handler.get_states(mode="tensor")
    obs_saver.add(obs_orin)

    for _ in range(250):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()
        obs_orin = env_inference.env.handler.get_states(mode="tensor")
        obs_saver.add(obs_orin)

    # obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
