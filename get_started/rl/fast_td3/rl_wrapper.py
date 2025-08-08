from __future__ import annotations

import numpy as np
import torch
from torchvision.utils import make_grid

from scenario_cfg.scenario import ScenarioCfg
from roboverse_learn.tasks.base import BaseTaskWrapper


class RLTaskWrapper(BaseTaskWrapper):
    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize the base class
        super().__init__(scenario)

        self.num_envs = scenario.num_envs
        self.robot = scenario.robots[0]

        # ----------- initial states --------------------------------------------------
        # Get initial states from the environment
        initial_states = self.env.get_initial_states() if hasattr(self.env, "get_initial_states") else []
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]

        # Reset environment with initial states
        self.reset(env_ids=list(range(self.num_envs)))

        # Get first observation to determine observation space
        states = self.env.get_states()
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]
        self._raw_observation_cache = first_obs.clone()

        # Set up action space
        limits = self.robot.joint_limits  # dict: {joint_name: (low, high)}
        self.joint_names = self.env.get_joint_names(self.robot.name)

        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self.num_actions = self._action_low.shape[0]

        self.max_episode_steps = 1000  # change this to the episode length of the task
        self.asymmetric_obs = False  # privileged critic input not used (for now)

    def _observation(self, env_states) -> torch.Tensor:
        """Flatten humanoid states and move them onto the training device."""
        if hasattr(self.env.task, "humanoid_obs_flatten_func"):
            return self.env.task.humanoid_obs_flatten_func(env_states).to(self.device)
        else:
            # Fallback: return flattened states
            return torch.tensor(env_states, device=self.device).flatten(start_dim=1)

    def _reward(self, env_states) -> torch.Tensor:
        """Get the reward of the environment."""
        total_reward = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self.env.task, "reward_functions") and hasattr(self.env.task, "reward_weights"):
            for reward_fn, weight in zip(self.env.task.reward_functions, self.env.task.reward_weights):
                total_reward += reward_fn(self.robot.name)(env_states).to(self.device) * weight
        else:
            # Fallback: return zero reward
            total_reward = torch.zeros(self.num_envs, device=self.device)
        return total_reward

    def _time_out(self, env_states) -> torch.Tensor:
        """Get the timeout flag of the environment based on max episode length."""
        timeout_flag = self._episode_steps >= self.max_episode_steps
        return timeout_flag

    def reset(self, env_ids) -> torch.Tensor:
        """Reset the environment and return initial observation."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0
        self.env.reset(states=self._initial_states, env_ids=env_ids)
        states = self.env.get_states()
        observation = self._observation(states)
        observation = observation.to(self.device)
        self._raw_observation_cache.copy_(observation)
        return observation

    def step(self, actions: torch.Tensor):
        """Step the environment with normalized actions."""
        self._episode_steps += 1

        # Convert normalized actions to real actions
        real_action = self._unnormalise_action(actions)

        # Use the base class step method
        obs, priv_obs, reward, terminated, time_out, _ = super().step(real_action)

        obs_now = obs.to(self.device)
        reward_now = reward.to(self.device)
        done_flag = terminated.to(self.device, torch.bool)
        time_out_flag = time_out.to(self.device, torch.bool)

        info = {
            "time_outs": time_out_flag,
            "episode_steps": self._episode_steps.clone(),
            "observations": {"raw": {"obs": self._raw_observation_cache.clone().to(self.device)}},
        }

        # Check for episode completion (either terminated or timed out)
        episode_done = done_flag | time_out_flag

        if (done_indices := episode_done.nonzero(as_tuple=False).squeeze(-1)).numel():
            # Reset completed episodes
            self.reset(env_ids=done_indices.tolist())
            reset_states = self.env.get_states()
            reset_obs_full = self._observation(reset_states).to(self.device)
            obs_now[done_indices] = reset_obs_full[done_indices]
            self._raw_observation_cache[done_indices] = reset_obs_full[done_indices]
        else:
            keep_mask = (~done_flag).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs_now)

        return obs_now, reward_now, episode_done, info

    def render(self) -> np.ndarray:
        """Render the environment and return RGB image."""
        state = self.env.get_states()
        rgb_data = next(iter(state.cameras.values())).rgb
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        return image

    def _unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions from [-1, 1] to the robot's joint-limit range."""
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low
