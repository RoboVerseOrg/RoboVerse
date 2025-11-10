from __future__ import annotations

import warnings
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import _find_spec, register
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.vector_env import VectorEnv

from .registry import get_task_class, list_tasks

# Local fallback registry for vector entry points when Gymnasium does not
# support the `vector_entry_point` argument in `register()`.
_VECTOR_ENTRY_POINTS: dict[str, Callable[..., VectorEnv]] = {}

# Use the official enum for autoreset mode (required to silence the warning)
try:
    from gymnasium.vector import AutoresetMode
except Exception:
    AutoresetMode = None  # Fallback won't silence the warning, but keeps compatibility


# -------------------------
# Single-env Gym wrapper (for gym.make)
# -------------------------
class GymEnvWrapper(gym.Env):
    """Gymnasium-compatible single-environment wrapper around the RL task."""

    # Render metadata (class-level defaults)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str,
        device: str | torch.device | None = None,
        **scenario_kwargs: Any,
    ) -> None:
        # Force single environment when created via gym.make.
        scenario_kwargs["num_envs"] = 1
        self.device = device
        self.task_cls = get_task_class(task_name)
        updated_scenario_cfg = self.task_cls.scenario.update(**scenario_kwargs)
        self.scenario = updated_scenario_cfg
        self.task_env = self.task_cls(updated_scenario_cfg, device)
        self.action_space = self.task_env.action_space
        self.observation_space = self.task_env.observation_space
        # Instance-level metadata; declare autoreset mode with the official enum
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["autoreset_mode"] = (
            AutoresetMode.SAME_STEP if AutoresetMode is not None else "same-step"
        )  # If enum missing, string fallback (may still warn on older Gymnasium)
        self.device = self.task_env.device

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        obs, info = self.task_env.reset()
        return obs, info

    def step(self, action):
        """Step the environment with the given action."""
        # Backend is expected to return (obs, reward, terminated, truncated, info).
        obs, reward, terminated, truncated, info = self.task_env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        img = self.task_env.render()
        # Return a safe copy in case the backend reuses buffers.
        return None if img is None else np.array(img, copy=True)

    def close(self):
        """Close the environment."""
        self.task_env.close()


# -------------------------
# VectorEnv adapter (native backend vectorization; for gym.make_vec)
# -------------------------
class GymVectorEnvAdapter(VectorEnv):
    """VectorEnv adapter that leverages backend-native vectorization (single process, many envs)."""

    # Render metadata (class-level defaults)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str,
        device: str | torch.device | None = None,
        **scenario_kwargs: Any,
    ) -> None:
        self.task_cls = get_task_class(task_name)
        updated_scenario_cfg = self.task_cls.scenario.update(**scenario_kwargs)
        self.task_env = self.task_cls(updated_scenario_cfg, device)
        self.scenario = updated_scenario_cfg
        self.device = self.task_env.device
        try:
            super().__init__(self.task_env.num_envs, self.task_env.observation_space, self.task_env.action_space)
        except TypeError:
            # Some versions may not define VectorEnv.__init__.
            self.num_envs = self.task_env.num_envs
            self.observation_space = self.task_env.observation_space
            self.action_space = self.task_env.action_space

        # Optional single-space hints consumed by some libraries.
        self.single_observation_space = self.task_env.observation_space
        self.single_action_space = self.task_env.action_space

        self._pending_actions: torch.Tensor | None = None

        # Instance-level metadata; declare autoreset mode with the official enum
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["autoreset_mode"] = AutoresetMode.SAME_STEP if AutoresetMode is not None else "same-step"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all environments and return initial observations."""
        obs, info = self.task_env.reset()
        return obs, info

    def step_async(self, actions) -> None:
        """Cache actions; convert to torch."""
        self._pending_actions = actions

    def step_wait(self):
        """Wait for the step to complete and return results."""
        obs, reward, terminated, truncated, info = self.task_env.step(self._pending_actions)
        self._pending_actions = None  # Clear pending actions.
        return obs, reward, terminated, truncated, info

    def step(self, actions):
        """Synchronous step composed from step_async + step_wait (required by Gym)."""
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        """Render the environment."""
        img = self.task_env.render()
        return None if img is None else np.array(img, copy=True)

    def close(self):
        """Close the environment."""
        self.task_env.close()


# -------------------------
# Entry points for registration
# -------------------------
def _make_entry_point_single(task_name: str) -> Callable[..., gym.Env]:
    """Entry point for gym.make(): always returns a single-env GymEnvWrapper."""

    def _factory(**kwargs: Any) -> gym.Env:
        device = kwargs.pop("device", None)
        # Ignore any external num_envs to keep gym.make() single-env.
        kwargs.pop("num_envs", None)
        return GymEnvWrapper(task_name=task_name, device=device, **kwargs)

    return _factory


def _make_vector_entry_point(task_name: str) -> Callable[..., VectorEnv]:
    """Entry point for gym.make_vec(): returns a native-vectorized VectorEnv."""

    def _factory(**kwargs: Any) -> VectorEnv:
        device = kwargs.pop("device", None)
        num_envs = int(kwargs.pop("num_envs", 1) or 1)
        prefer_backend_vectorization = bool(kwargs.pop("prefer_backend_vectorization", True))

        # Optional fallback to SyncVectorEnv for non-native backends or debugging.
        if not prefer_backend_vectorization and num_envs > 1:

            def _one_env_factory():
                return GymEnvWrapper(task_name=task_name, device=device, **kwargs)

            return SyncVectorEnv([_one_env_factory for _ in range(num_envs)])

        return GymVectorEnvAdapter(task_name=task_name, num_envs=num_envs, device=device, **kwargs)

    return _factory


# -------------------------
# Registration helpers
# -------------------------
def register_all_tasks_with_gym(prefix: str = "RoboVerse/") -> None:
    """Register all known tasks for both gymnasium.make and gymnasium.make_vec.

    This is safe to call multiple times.
    """
    for task_name in list_tasks():
        env_id = f"{prefix}{task_name}"
        entry = _make_entry_point_single(task_name)
        vec_entry = _make_vector_entry_point(task_name)
        try:
            register(id=env_id, entry_point=entry, vector_entry_point=vec_entry)
        except TypeError:
            try:
                register(id=env_id, entry_point=entry)
            except Exception:
                pass
            _VECTOR_ENTRY_POINTS[env_id] = vec_entry
            try:
                spec_ = _find_spec(env_id)
                spec_.vector_entry_point = vec_entry  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            # Ignore duplicate registrations during hot reload.
            pass


def register_task_with_gym(task_name: str, env_id: str | None = None) -> str:
    """Register a single task with both single-env and vectorized entry points."""
    if env_id is None:
        env_id = f"RoboVerse/{task_name}"
    entry = _make_entry_point_single(task_name)
    vec_entry = _make_vector_entry_point(task_name)
    try:
        register(id=env_id, entry_point=entry, vector_entry_point=vec_entry)
    except TypeError:
        try:
            # Older Gymnasium versions do not support vector_entry_point as a kwarg.
            # Register the single-env entry first.
            register(id=env_id, entry_point=entry)
        except Exception:
            # Ignore duplicate registrations during hot reload.
            pass
        # Store locally for our make_vec fallback.
        _VECTOR_ENTRY_POINTS[env_id] = vec_entry
        # Additionally, attach the vector entry point to the spec if available so
        # gymnasium.make_vec can discover it on older versions.
        try:
            spec_ = _find_spec(env_id)
            # Some Gymnasium versions allow arbitrary attributes on EnvSpec.
            # Use direct attribute assignment to avoid linter warnings about setattr.
            spec_.vector_entry_point = vec_entry  # type: ignore[attr-defined]
        except Exception:
            # Best-effort: make_vec helper will still work using our local registry.
            pass
    except Exception:
        # Ignore duplicate registrations during hot reload.
        pass
    return env_id


def make_vec(
    env_id: str,
    num_envs: int,
    **kwargs: Any,
) -> VectorEnv:
    """Deprecated: use gymnasium.make_vec(env_id, num_envs=..., ...) instead.

    This wrapper will attempt to ensure the Gymnasium spec has a vector entry
    point, register it if missing, then forward the call to gymnasium.make_vec.
    """
    warnings.warn(
        "metasim.task.gym_registration.make_vec is deprecated; use gymnasium.make_vec instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Ensure vector entry point is available for env_id
    try:
        spec_ = _find_spec(env_id)
        vec_ep = getattr(spec_, "vector_entry_point", None)
    except Exception:
        spec_ = None
        vec_ep = None

    if vec_ep is None:
        # Try to register this specific task lazily
        task_name = env_id.split("/", 1)[1] if "/" in env_id else env_id
        register_task_with_gym(task_name, env_id=env_id)

    # Forward to Gymnasium
    return gym.make_vec(env_id, num_envs=num_envs, **kwargs)
