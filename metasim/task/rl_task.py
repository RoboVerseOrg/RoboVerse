from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torchvision.utils import make_grid

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.types import CompatActionInput, Info
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
        # ``self.robots`` (list) and ``self.robot`` (first, scalar) are both
        # part of the contract subclasses rely on — keep them in sync here.
        self.robots: list = list(scenario.robots) if scenario.robots else []
        self.robot = self.robots[0] if self.robots else None
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Observation buffers for RSL-RL compatibility
        self._obs_buf = None
        self._priv_obs_buf = None

        # convert list state to tensor state for reset acceleration
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)
        # first reset
        self.reset(env_ids=list(range(self.num_envs)))

        # obs size
        states = self.handler.get_states(mode="tensor")
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]

        # action bounds from joint limits in handler/API tensor order
        self.joint_names_by_robot = {
            robot.name: self.handler.get_joint_names(robot.name, sort=True) for robot in self.robots
        }
        action_low = []
        action_high = []
        for robot in self.robots:
            limits = robot.joint_limits
            joint_names = self.joint_names_by_robot[robot.name]
            action_low.extend(limits[j][0] for j in joint_names)
            action_high.extend(limits[j][1] for j in joint_names)
        self._action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self._action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)
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
    def reset(self, states=None, env_ids=None, seed: int | None = None) -> tuple[torch.Tensor, Info]:
        """Reset selected envs.

        Args:
            env_ids: Indices to reset; None resets all.
            states: Optional external states to set for the selected envs. If None, use initial states.
            seed: Optional reproducibility seed forwarded to ``handler.set_seed`` when
                the backend implements it. See ``TaskBase.reset`` for the warn-if-
                unsupported semantics.

        Returns:
            (obs, info).
        """
        if seed is not None:
            set_seed = getattr(self.handler, "set_seed", None)
            if callable(set_seed):
                set_seed(seed)
            elif not getattr(self, "_seed_unsupported_warned", False):
                from loguru import logger as _log

                _log.warning(
                    f"{type(self).__name__}: handler "
                    f"{type(self.handler).__name__} does not implement set_seed; "
                    f"reset(seed={seed}) is a no-op on the simulator side."
                )
                self._seed_unsupported_warned = True
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0
        raw_states = self._initial_states if states is None else states
        states_to_set = self._prepare_states(raw_states, env_ids)
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        states = self.handler.get_states(mode="tensor")
        first_obs = self._observation(states).to(self.device)
        self._raw_observation_cache = first_obs.clone()
        priv_obs = self._privileged_observation(states)

        # Update observation buffers for RSL-RL compatibility
        self._obs_buf = first_obs
        if isinstance(priv_obs, torch.Tensor):
            self._priv_obs_buf = priv_obs.to(self.device)
        else:
            self._priv_obs_buf = first_obs

        info = {"privileged_observation": priv_obs}
        return first_obs, info

    def step(
        self,
        actions: CompatActionInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Info]:
        """One step with joint-space actions (auto-clamped)."""
        self._episode_steps += 1

        if isinstance(actions, (torch.Tensor, np.ndarray)):
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)

            real_actions = torch.maximum(torch.minimum(actions, self._action_high), self._action_low)
            self.handler.set_dof_targets(real_actions)
        else:
            self.handler.set_dof_targets(actions)
        self.handler.simulate()
        states = self.handler.get_states(mode="tensor")
        obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states)
        # Cross-backend contract: normalise reward to (num_envs,) float32 on
        # self.device. Individual ``_reward`` implementations may return a
        # tensor on whatever device / dtype get_states produced
        # (e.g. MuJoCo CPU vs Newton CUDA), which then leaked into the
        # gym-side reward tuple — same task → different reward dtype per
        # backend. Forcing the cast here keeps the public step() contract
        # uniform without requiring every task to remember to convert.
        reward = self._reward(states).to(device=self.device, dtype=torch.float32)
        terminated = self._terminated(states).bool().to(self.device)
        time_out = self._time_out(states).bool().to(self.device)

        # Cache observations for RSL-RL compatibility
        self._obs_buf = obs
        if isinstance(priv_obs, torch.Tensor):
            self._priv_obs_buf = priv_obs.to(self.device)
        else:
            self._priv_obs_buf = obs

        episode_done = terminated | time_out
        info = {
            "privileged_observation": priv_obs,
            "episode_steps": self._episode_steps.clone(),
            "observations": {"raw": {"obs": self._raw_observation_cache.clone()}},
        }

        done_indices = episode_done.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel():
            self.reset(env_ids=done_indices.tolist())
            states_after = self.handler.get_states(mode="tensor")
            obs_after = self._observation(states_after).to(self.device)
            obs[done_indices] = obs_after[done_indices]
            self._raw_observation_cache[done_indices] = obs_after[done_indices]
        else:
            keep_mask = (~terminated).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs)

        return obs, reward, terminated, time_out, info

    def render(self) -> np.ndarray:
        """Return an RGB grid image."""
        state = self.handler.get_states(mode="tensor")
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

    # -------------------------------------------------------------------------
    # RSL-RL compatibility properties
    # -------------------------------------------------------------------------

    @property
    def obs_buf(self) -> torch.Tensor:
        """Cached observation buffer for RSL-RL compatibility.

        This property enables RLTaskEnv-based environments to work with
        RSL-RL's OnPolicyRunner without needing a wrapper.
        """
        if self._obs_buf is None:
            # Lazy initialization on first access
            states = self.handler.get_states(mode="tensor")
            self._obs_buf = self._observation(states).to(self.device)
        return self._obs_buf

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Cached privileged observation buffer for RSL-RL compatibility.

        Returns privileged observations if available, otherwise returns
        the same as obs_buf (symmetric actor-critic).
        """
        if self._priv_obs_buf is None:
            # Lazy initialization on first access
            states = self.handler.get_states(mode="tensor")
            priv_obs = self._privileged_observation(states)
            if isinstance(priv_obs, torch.Tensor):
                self._priv_obs_buf = priv_obs.to(self.device)
            else:
                # Fallback to symmetric observations
                self._priv_obs_buf = self.obs_buf
        return self._priv_obs_buf
