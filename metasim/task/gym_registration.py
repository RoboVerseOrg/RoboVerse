from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import _find_spec, register
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.vector_env import VectorEnv

from .registry import get_task_class

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

        self._device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        self.task_cls = get_task_class(task_name)
        updated_scenario_cfg = self.task_cls.scenario.update(**scenario_kwargs)
        self.scenario = updated_scenario_cfg
        self.env = self.task_cls(updated_scenario_cfg)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Instance-level metadata; declare autoreset mode with the official enum
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["autoreset_mode"] = (
            AutoresetMode.SAME_STEP if AutoresetMode is not None else "same-step"
        )  # If enum missing, string fallback (may still warn on older Gymnasium)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        obs, info = self.env.reset()

        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs[0]
        return obs, {}

    def step(self, action):
        """Step the environment with the given action."""
        # Three cases: numpy -> tensor, list-of-dict -> stacked tensor, torch -> move to device
        if isinstance(action, torch.Tensor):
            action_t = action.to(self._device)
        elif isinstance(action, np.ndarray):
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        elif isinstance(action, list):
            robot = self.scenario.robots[0]
            joint_names = list(robot.joint_limits.keys())

            if len(action) != 1:
                raise ValueError(f"Single-env wrapper expects exactly 1 action dict, got {len(action)}")

            vec = torch.tensor(
                [action[0][robot.name]["dof_pos_target"][jn] for jn in joint_names],
                dtype=torch.float32,
                device=self._device,
            )

            action_t = vec.unsqueeze(0)

        else:
            raise TypeError(
                f"Unsupported action type: {type(action)}. Expected torch.Tensor, numpy.ndarray, or list/dict of action dicts."
            )

        # Ensure batch dimension for single-env backend.
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)

        # Backend is expected to return (obs, reward, terminated, truncated, info).
        obs, reward, terminated, truncated, info = self.env.step(action_t)

        # De-batch observation for single-env wrapper if needed.
        try:
            if hasattr(obs, "ndim") and obs.ndim >= 2 and obs.shape[0] == 1:
                obs = obs[0]
        except Exception:
            pass

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        img = self.env.render()
        # Return a safe copy in case the backend reuses buffers.
        return None if img is None else np.array(img, copy=True)

    def close(self):
        """Close the environment."""
        self.env.close()


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
        num_envs: int,
        device: str | torch.device | None = None,
        **scenario_kwargs: Any,
    ) -> None:
        # Delegate num_envs to the backend.
        scenario_kwargs["num_envs"] = int(num_envs)

        self._device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        self.task_cls = get_task_class(task_name)
        updated_scenario_cfg = self.task_cls.scenario.update(**scenario_kwargs)
        self.env = self.task_cls(updated_scenario_cfg)
        self.scenario = updated_scenario_cfg
        # scenario = ScenarioCfg(**scenario_kwargs)
        # self.env = load_task(task_name, scenario, device=self._device)
        # self.scenario = self.env.scenario
        # Use positional args to be compatible across Gymnasium versions.
        try:
            super().__init__(self.env.num_envs, self.env.observation_space, self.env.action_space)
        except TypeError:
            # Some versions may not define VectorEnv.__init__.
            self.num_envs = self.env.num_envs
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

        # Optional single-space hints consumed by some libraries.
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space

        self._pending_actions: torch.Tensor | None = None

        # Instance-level metadata; declare autoreset mode with the official enum
        self.metadata = dict(getattr(self, "metadata", {}))
        self.metadata["autoreset_mode"] = AutoresetMode.SAME_STEP if AutoresetMode is not None else "same-step"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all environments and return initial observations."""
        # super().reset(seed=seed)
        obs, info = self.env.reset()
        # VectorEnv API: return (obs_batch, infos_list) where len(infos) == num_envs.
        return obs, info

    def step_async(self, actions) -> None:
        """Cache actions; convert to torch."""
        # Three cases: numpy -> tensor, list-of-dict -> stacked tensor, torch -> move to device
        if isinstance(actions, torch.Tensor):
            self._pending_actions = actions.to(self._device)
        elif isinstance(actions, np.ndarray):
            self._pending_actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        elif isinstance(actions, list):
            robot = self.scenario.robots[0]
            joint_names = list(robot.joint_limits.keys())
            if len(actions) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} action dicts, got {len(actions)}")
            # Build a clean tensor directly without intermediate lists of tensors
            act_dim = len(joint_names)
            actions_tensor = torch.empty((self.num_envs, act_dim), dtype=torch.float32, device=self._device)
            for env_index in range(self.num_envs):
                actions_tensor[env_index] = torch.tensor(
                    [actions[env_index][robot.name]["dof_pos_target"][jn] for jn in joint_names],
                    dtype=torch.float32,
                )
            self._pending_actions = actions_tensor
        else:
            raise TypeError(
                f"Unsupported action type: {type(actions)}. Expected torch.Tensor, numpy.ndarray, or list/dict of action dicts."
            )

    def step_wait(self):
        """Wait for the step to complete and return results."""
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait.")

        out = self.env.step(self._pending_actions)
        if len(out) != 5:
            raise RuntimeError(
                f"Backend returned {len(out)} items; expected 5 (obs, reward, terminated, truncated, info)."
            )

        obs, reward, terminated, truncated, info = out  # 'truncated' may be 'timeout' internally.

        # Ensure infos is a list[dict] of length num_envs.
        if info is None:
            infos = [{} for _ in range(self.num_envs)]
        elif isinstance(info, dict):
            infos = [info.copy() for _ in range(self.num_envs)]
        else:
            infos = list(info)
            if len(infos) != self.num_envs:
                # Broadcast a single dict if needed.
                if len(infos) == 1 and isinstance(infos[0], dict):
                    infos = [infos[0].copy() for _ in range(self.num_envs)]
                else:
                    raise RuntimeError(f"Expected {self.num_envs} infos, got {len(infos)}.")

        # Clear pending actions.
        self._pending_actions = None
        return obs, reward, terminated, truncated, infos

    def step(self, actions):
        """Synchronous step composed from step_async + step_wait (required by Gym)."""
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        """Render the environment."""
        img = self.env.render()
        return None if img is None else np.array(img, copy=True)

    def close(self):
        """Close the environment."""
        self.env.close()


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


# # -------------------------
# # Registration helpers
# # -------------------------
# def register_all_tasks_with_gym(prefix: str = "RoboVerse/") -> None:
#     """Register all tasks with both single-env and vectorized entry points."""
#     for task_name in list_tasks():
#         env_id = f"{prefix}{task_name}"
#         entry = _make_entry_point_single(task_name)
#         vec_entry = _make_vector_entry_point(task_name)
#         # Try registering with vector entry point (newer Gymnasium). If that fails
#         # (older versions), register single entry and store vector entry locally.
#         try:
#             register(id=env_id, entry_point=entry, vector_entry_point=vec_entry)
#         except TypeError:
#             try:
#                 register(id=env_id, entry_point=entry)
#             except Exception:
#                 # Ignore duplicate registrations during hot reload.
#                 pass
#             _VECTOR_ENTRY_POINTS[env_id] = vec_entry
#         except Exception:
#             # Ignore duplicate registrations during hot reload.
#             pass


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
            register(id=env_id, entry_point=entry)
        except Exception:
            # Ignore duplicate registrations during hot reload.
            pass
        _VECTOR_ENTRY_POINTS[env_id] = vec_entry
    except Exception:
        # Ignore duplicate registrations during hot reload.
        pass
    return env_id


def make_vec(
    env_id: str,
    num_envs: int,
    **kwargs: Any,
) -> VectorEnv:
    """Instantiate a vectorized roboverse task.

    Args:
        env_id: The environment ID to register.
        num_envs: The number of environments to create.
        **kwargs: Additional keyword arguments to pass to the environment creator.

    Returns:
        VectorEnv: The vectorized environment.
    """
    # Prefer locally stored vector entry (for older Gymnasium without
    # vector_entry_point support). Fall back to spec, then to building a
    # vector adapter directly from the task name.
    env_creator: Callable[..., VectorEnv] | None = _VECTOR_ENTRY_POINTS.get(env_id)

    if env_creator is None:
        try:
            spec_ = _find_spec(env_id)
            env_creator = getattr(spec_, "vector_entry_point", None)
        except Exception:
            env_creator = None

    if env_creator is None:
        # Derive task name from the env_id and build a vector entry on the fly.
        task_name = env_id.split("/", 1)[1] if "/" in env_id else env_id
        env_creator = _make_vector_entry_point(task_name)

    env = env_creator(num_envs=num_envs, **kwargs)
    return env
