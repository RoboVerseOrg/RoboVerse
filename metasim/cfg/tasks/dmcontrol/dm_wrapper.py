"""DM Control wrapper for RoboVerse integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from dm_control import suite, viewer
    from dm_env import specs

    DM_CONTROL_AVAILABLE = True
except ImportError:
    DM_CONTROL_AVAILABLE = False
    suite = None
    viewer = None
    specs = None


class DMControlWrapper:
    """Wrapper for dm_control environments to work with RoboVerse."""

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        num_envs: int = 1,
        time_limit: float | None = None,
        visualize_reward: bool = False,
        environment_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        headless: bool = True,
    ):
        """Initialize the dm_control wrapper.

        Args:
            domain_name: The domain name (e.g., 'walker', 'cartpole')
            task_name: The task name (e.g., 'walk', 'balance')
            num_envs: Number of parallel environments
            time_limit: Time limit for episodes
            visualize_reward: Whether to visualize rewards
            environment_kwargs: Additional environment kwargs
            random_state: Random seed
            headless: Whether to run in headless mode (no visualization)
        """
        if not DM_CONTROL_AVAILABLE:
            raise ImportError("dm_control is not installed. Please install it with: pip install dm-control")

        self.domain_name = domain_name
        self.task_name = task_name
        self.num_envs = num_envs
        self.headless = headless
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For visualization
        self._viewer = None
        self._render_every_frame = not headless
        self._viewer_running = False

        # Create environments
        self.envs = []
        self.time_steps = []

        for i in range(num_envs):
            kwargs = {"domain_name": domain_name, "task_name": task_name, "visualize_reward": visualize_reward}

            if time_limit is not None:
                kwargs["task_kwargs"] = {"time_limit": time_limit}

            if environment_kwargs:
                kwargs["environment_kwargs"] = environment_kwargs

            if random_state is not None:
                kwargs["task_kwargs"] = kwargs.get("task_kwargs", {})
                kwargs["task_kwargs"]["random"] = random_state + i

            env = suite.load(**kwargs)
            self.envs.append(env)
            self.time_steps.append(None)

        # Get action and observation specs from first environment
        self.action_spec = self.envs[0].action_spec()
        self.observation_spec = self.envs[0].observation_spec()

        # Calculate observation and action dimensions
        self.obs_dim = self._calculate_obs_dim()
        self.action_dim = np.prod(self.action_spec.shape)

        # Episode tracking
        self._episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

    def _calculate_obs_dim(self) -> int:
        """Calculate total observation dimension."""
        obs_dim = 0
        for key, spec in self.observation_spec.items():
            if hasattr(spec, "shape"):
                obs_dim += np.prod(spec.shape)
        return obs_dim

    def reset(self, env_ids: list[int] | None = None) -> torch.Tensor:
        """Reset specified environments.

        Args:
            env_ids: List of environment IDs to reset. If None, reset all.

        Returns:
            Observations as a torch tensor of shape (num_envs, obs_dim)
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        observations = []

        for i in range(self.num_envs):
            if i in env_ids:
                self.time_steps[i] = self.envs[i].reset()
                self._episode_length_buf[i] = 0

            # Get observation
            obs = self._flatten_observation(self.time_steps[i].observation if self.time_steps[i] else {})
            observations.append(obs)

        return torch.stack(observations).to(self.device)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments with given actions.

        Args:
            actions: Actions as a torch tensor of shape (num_envs, action_dim)

        Returns:
            Tuple of (observations, rewards, dones, timeouts)
        """
        # Convert actions to numpy
        actions_np = actions.cpu().numpy()

        observations = []
        rewards = []
        dones = []

        # Step each environment
        for i in range(self.num_envs):
            if self.time_steps[i] is None:
                # Environment not initialized, reset it
                self.time_steps[i] = self.envs[i].reset()

            # Step environment
            self.time_steps[i] = self.envs[i].step(actions_np[i])
            self._episode_length_buf[i] += 1

            # Get observation
            obs = self._flatten_observation(self.time_steps[i].observation)
            observations.append(obs)

            # Get reward
            reward = self.time_steps[i].reward or 0.0
            rewards.append(reward)

            # Check if done
            done = self.time_steps[i].last()
            dones.append(done)

        # Convert to tensors
        obs_tensor = torch.stack(observations).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Timeouts are handled separately in RoboVerse
        timeout_tensor = torch.zeros_like(done_tensor)

        # Update viewer if running
        if self._viewer_running and hasattr(self, "_viewer") and self._viewer is not None:
            try:
                self._viewer.sync()
            except:
                pass  # Ignore viewer errors

        return obs_tensor, reward_tensor, done_tensor, timeout_tensor

    def _flatten_observation(self, obs_dict: dict[str, np.ndarray]) -> torch.Tensor:
        """Flatten observation dictionary into a single tensor."""
        if not obs_dict:
            return torch.zeros(self.obs_dim, dtype=torch.float32)

        obs_list = []
        for key in sorted(obs_dict.keys()):
            obs = obs_dict[key]
            if isinstance(obs, np.ndarray):
                obs_list.append(obs.flatten())
            else:
                obs_list.append(np.array([obs]))

        flat_obs = np.concatenate(obs_list)
        return torch.tensor(flat_obs, dtype=torch.float32)

    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def render(self, mode="rgb_array"):
        """Render the first environment."""
        if not self.envs:
            return None

        # If we're in headless mode, just return the RGB array
        if self.headless or mode == "rgb_array":
            if hasattr(self.envs[0], "render"):
                return self.envs[0].render(mode="rgb_array")
            return None

        # For non-headless mode, we need to use mujoco's viewer
        # This requires accessing the mujoco model directly
        if hasattr(self.envs[0], "physics") and hasattr(self.envs[0].physics, "render"):
            # Get camera view
            return self.envs[0].physics.render(camera_id=0)

        return None

    def launch_viewer(self):
        """Launch interactive viewer for the first environment."""
        if not self.headless:
            try:
                # Try to use mujoco's native viewer
                import mujoco.viewer

                # Get the mujoco model and data from the first environment
                if hasattr(self.envs[0], "physics"):
                    physics = self.envs[0].physics
                    if hasattr(physics, "model") and hasattr(physics, "data"):
                        self._mj_model = physics.model._model
                        self._mj_data = physics.data._data

                        # Launch mujoco viewer in passive mode
                        self._viewer = mujoco.viewer.launch_passive(
                            self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=True
                        )
                        self._viewer_running = True
                        return

                # Fallback to dm_control viewer if available
                if viewer is not None:
                    # Create a policy that returns the last action
                    self._last_action = np.zeros(self.action_dim)

                    def policy(time_step):
                        return self._last_action

                    # Launch viewer in a separate thread
                    import threading

                    viewer_thread = threading.Thread(
                        target=viewer.launch, args=(self.envs[0],), kwargs={"policy": policy}
                    )
                    viewer_thread.daemon = True
                    viewer_thread.start()
                    self._viewer_running = True
            except Exception as e:
                pass

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Get episode length buffer."""
        return self._episode_length_buf
