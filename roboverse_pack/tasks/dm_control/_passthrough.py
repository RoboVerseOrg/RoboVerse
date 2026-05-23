"""dm_control passthrough — wrap suite envs as gymnasium envs under DMControl/<domain>/<task>."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger as log


def register_dm_control_passthrough(prefix: str = "DMControl/") -> list[str]:
    """Register all dm_control suite tasks under DMControl/<domain>-<task> namespace."""
    from gymnasium.envs.registration import register, registry

    import dm_control.suite as suite

    registered = []
    failed = []
    for domain, task in suite.ALL_TASKS:
        ns_id = f"{prefix}{domain}-{task}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.dm_control._passthrough:_make_dm_env",
                kwargs={"domain": domain, "task": task},
            )
            registered.append(ns_id)
        except Exception as e:
            failed.append((ns_id, str(e)))
    if failed:
        log.warning(f"[dm_control_passthrough] {len(failed)} failures")
    return registered


import gymnasium as gym


class _DMControlGymWrapper(gym.Env):
    """Minimal dm_control TimeStep → gymnasium adapter."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, domain: str, task: str, **kwargs):
        super().__init__()
        from dm_control import suite

        self._env = suite.load(
            domain_name=domain, task_name=task, visualize_reward=kwargs.pop("visualize_reward", False)
        )
        spec = self._env.observation_spec()
        # Concatenate all obs into a single Box (typical practice for dm_control)
        import gymnasium as gym

        self._obs_keys = sorted(spec.keys())
        obs_dim = sum(int(np.prod(spec[k].shape)) for k in self._obs_keys)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        a_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=a_spec.minimum.astype(np.float32),
            high=a_spec.maximum.astype(np.float32),
            shape=a_spec.shape,
            dtype=np.float32,
        )
        self._domain = domain
        self._task = task

    def _flatten_obs(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._obs_keys])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.task.random.seed(seed)
        ts = self._env.reset()
        return self._flatten_obs(ts.observation), {}

    def step(self, action):
        ts = self._env.step(action)
        terminated = ts.last() and ts.discount == 0.0
        truncated = ts.last() and not terminated
        return self._flatten_obs(ts.observation), float(ts.reward or 0.0), terminated, truncated, {}

    def render(self):
        return self._env.physics.render(height=360, width=640, camera_id=0)

    def close(self):
        self._env.close()


def _make_dm_env(domain: str, task: str, **kwargs: Any):
    return _DMControlGymWrapper(domain, task, **kwargs)
