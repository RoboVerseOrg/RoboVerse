from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torchvision.utils import make_grid

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.state import list_state_to_tensor


class RLTaskEnv(BaseTaskEnv):
    """Common utilities for RL tasks."""

    max_episode_steps = 1000

    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize environment."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._observation_space: spaces.Space | None = None
        self._action_space: spaces.Space | None = None

        self.asymmetric_obs = False

        super().__init__(scenario, device)

        self.num_envs = scenario.num_envs
        self.robot = scenario.robots[0]
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # convert list state to tensor state for reset acceleration
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)
        # first reset
        self.reset(env_ids=list(range(self.num_envs)))

        # obs size
        states = self.handler.get_states()
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]

        # action bounds from joint limits (ordered by joint_names)
        limits = self.robot.joint_limits
        self.joint_names = self.handler.get_joint_names(self.robot.name)
        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self.num_actions = self._action_low.shape[0]

    # -------------------------------------------------------------------------
    # hooks / spaces
    # -------------------------------------------------------------------------

    def _get_initial_states(self) -> list[dict]:
        """Return per-env initial states (override in subclasses)."""
        return None  # base expects subclass override

    @property
    def observation_space(self) -> spaces.Space:
        """Observation Box(num_obs,)."""
        if self._observation_space is None:
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_obs,),
                dtype=np.float32,
            )
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Action Box(num_actions,) with range [-1, 1]."""
        if self._action_space is None:
            self._action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_actions,),
                dtype=np.float32,
            )
        return self._action_space

    # -------------------------------------------------------------------------
    # env api
    # -------------------------------------------------------------------------
    def reset(self, states=None, env_ids=None) -> tuple[torch.Tensor, dict]:
        """Reset selected envs.

        Args:
            env_ids: Indices to reset; None resets all.
            states: Optional external states to set for the selected envs. If None, use initial states.

        Returns:
            (obs, info).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0
        raw_states = self._initial_states if states is None else states
        states_to_set = self._prepare_states(raw_states, env_ids)
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        states = self.handler.get_states()
        first_obs = self._observation(states).to(self.device)
        self._raw_observation_cache = first_obs.clone()
        info = {"privileged_observation": self._privileged_observation(states)}
        return first_obs, info

    def step(
        self,
        actions,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """One step with joint-space actions (auto-clamped)."""
        self._episode_steps += 1

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        real_actions = torch.maximum(torch.minimum(actions, self._action_high), self._action_low)
        self.handler.set_dof_targets(real_actions)
        self.handler.simulate()
        states = self.handler.get_states()
        obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states)
        reward = self._reward(states)
        terminated = self._terminated(states).bool().to(self.device)
        time_out = self._time_out(states).bool().to(self.device)

        episode_done = terminated | time_out
        info = {
            "privileged_observation": priv_obs,
            "episode_steps": self._episode_steps.clone(),
            "observations": {"raw": {"obs": self._raw_observation_cache.clone()}},
        }

        done_indices = episode_done.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel():
            self.reset(env_ids=done_indices.tolist())
            states_after = self.handler.get_states()
            obs_after = self._observation(states_after).to(self.device)
            obs[done_indices] = obs_after[done_indices]
            self._raw_observation_cache[done_indices] = obs_after[done_indices]
        else:
            keep_mask = (~terminated).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs)

        return obs, reward, terminated, time_out, info

    def render(self) -> np.ndarray:
        """Return an RGB grid image."""
        state = self.handler.get_states()
        rgb = next(iter(state.cameras.values())).rgb  # (N, H, W, C)
        if make_grid is not None:
            grid = make_grid((rgb.permute(0, 3, 1, 2) / 255.0), nrow=int(max(1, rgb.shape[0] ** 0.5)))
            return (grid.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return rgb[0].cpu().numpy().astype(np.uint8)

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] to joint limits."""
        return (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low

    def _reward(self, env_states) -> torch.Tensor:
        """Weighted sum of reward terms."""
        total_reward = None
        if len(self.reward_functions) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for reward_func, weight in zip(self.reward_functions, self.reward_weights):
            val = reward_func(env_states)
            if total_reward is None:
                total_reward = torch.zeros_like(val)
            total_reward += weight * val
        return total_reward

    def _terminated(self, env_states) -> torch.Tensor:
        """Terminal flags (default: none)."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _prepare_states(self, env_states, env_ids) -> torch.Tensor:
        """Prepare for the states before reset(do domain randomization)."""
        return env_states
