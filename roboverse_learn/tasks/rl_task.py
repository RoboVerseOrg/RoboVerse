from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torchvision.utils import make_grid

from roboverse_learn.tasks.base import BaseTaskWrapper
from scenario_cfg.scenario import ScenarioCfg


class RLTaskWrapper(BaseTaskWrapper):
    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # allow subclasses to tweak scenario before base init
        self._load_task_config(scenario)
        super().__init__(scenario)

        # basic handles
        self.num_envs = scenario.num_envs
        self.robot = scenario.robots[0]
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # initial states (replicate to num_envs)
        initial_states = self._get_initial_states()
        if len(initial_states) < self.num_envs:
            n = len(initial_states)
            reps, extra = divmod(self.num_envs, n)
            initial_states = initial_states * reps + initial_states[:extra]
        self._initial_states = initial_states[: self.num_envs]

        # reset all envs
        self.reset(env_ids=list(range(self.num_envs)))

        # observation size from first obs
        states = self.env.get_states()
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]

        # action bounds from joint limits (ordered by joint_names)
        limits = self.robot.joint_limits
        self.joint_names = self.env._get_joint_names(self.robot.name)
        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.num_actions = self._action_low.shape[0]

        # i/o dims for convenience
        self.input_dim = self.num_obs
        self.output_dim = self.num_actions

        # lazily built spaces
        self._observation_space = None
        self._action_space = None

        # episode settings
        self.max_episode_steps = 1000

        # asymmetric observation space
        self.asymmetric_obs = False

    # -------------------------------------------------------------------------
    # hooks / spaces
    # -------------------------------------------------------------------------

    def _load_task_config(self, scenario: ScenarioCfg) -> ScenarioCfg:
        """Hook to modify `scenario` before base init if needed."""
        return scenario

    def _get_initial_states(self) -> list[dict]:
        """Return initial states for each env (override in subclasses)."""
        return None  # intentional: base class expects subclass override

    @property
    def observation_space(self) -> spaces.Space:
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
        # normalized action space
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

    def _time_out(self, env_states) -> torch.Tensor:
        """Timeout flag based on max episode length."""
        return self._episode_steps >= self.max_episode_steps

    def reset(self, env_ids=None) -> torch.Tensor:
        """Reset and return initial observation."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0
        self.env.set_states(states=self._initial_states, env_ids=env_ids)

        states = self.env.get_states()
        first_obs = self._observation(states).to(self.device)
        # initialise cache on first reset
        self._raw_observation_cache = first_obs.clone()
        return first_obs

    def step(self, actions):
        """Step with normalized actions in [-1, 1]."""
        self._episode_steps += 1

        real_action = self._unnormalise_action(actions)

        self.env.set_dof_targets(self.robot.name, real_action)
        self.env.simulate()

        states = self.env.get_states()
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
            # reset finished envs and replace their obs
            self.reset(env_ids=done_indices.tolist())
            states_after = self.env.get_states()
            obs_after = self._observation(states_after).to(self.device)
            obs[done_indices] = obs_after[done_indices]
            self._raw_observation_cache[done_indices] = obs_after[done_indices]
        else:
            # keep previous raw obs where not done (same as original logic)
            keep_mask = (~terminated).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs)

        return obs, reward, terminated, time_out, info

    def render(self) -> np.ndarray:
        """Render RGB image grid from first camera."""
        state = self.env.get_states()
        rgb = next(iter(state.cameras.values())).rgb  # (N, H, W, C)
        grid = make_grid(  # (C, H, W)
            (rgb.permute(0, 3, 1, 2) / 255.0),
            nrow=int(rgb.shape[0] ** 0.5),
        )
        img = (grid.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return img

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def _unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions from [-1, 1] to the robot joint range."""
        return (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low

    def _reward(self, env_states) -> torch.Tensor:
        """Weighted sum of configured reward functions."""
        total_reward = None
        for reward_func, weight in zip(self.reward_functions, self.reward_weights):
            val = reward_func(env_states, self.robot.name)
            if total_reward is None:
                total_reward = torch.zeros_like(val)
            total_reward += weight * val
        return total_reward

    def _terminated(self, env_states) -> torch.Tensor:
        """Check if any envs are terminated."""
        # default: no termination
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
