from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import rootutils
import torch
from gymnasium.envs.registration import register

# Ensure project root is on sys.path when running scripts directly
rootutils.setup_root(__file__, pythonpath=True)

from importlib import import_module
from pathlib import Path

from scenario_cfg.scenario import ScenarioCfg

from .registry import list_tasks, load_task


class GymEnvWrapper(gym.Env):
    """Gymnasium-compatible wrapper around RL task wrapper.

    This wrapper adapts the step/reset signatures and converts tensors to numpy
    so it can be created via gym.make(). It is intended for single-env usage
    (num_envs == 1). For vectorized training, prefer SB3 VecEnv on the original
    RL wrapper.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_name: str,
        device: str | torch.device | None = None,
        **scenario_kwargs: Any,
    ) -> None:
        if "num_envs" not in scenario_kwargs:
            scenario_kwargs["num_envs"] = 1

        self._device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        scenario = ScenarioCfg(**scenario_kwargs)
        self.env = load_task(task_name, scenario, device=self._device)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        obs = self.env.reset()
        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        return obs, {}

    def step(self, action: np.ndarray):
        """Step the environment with the given action."""
        if not isinstance(action, torch.Tensor):
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        else:
            action_t = action.to(self._device, dtype=torch.float32)

        # ensure batch dim for single env
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)

        obs, reward, terminated, time_out, info = self.env.step(action_t)

        # convert
        if torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if torch.is_tensor(reward):
            reward = float(reward.detach().cpu().numpy().reshape(-1)[0])
        if torch.is_tensor(terminated):
            terminated = bool(terminated.detach().cpu().numpy().reshape(-1)[0])
        if torch.is_tensor(time_out):
            time_out = bool(time_out.detach().cpu().numpy().reshape(-1)[0])

        return obs[0], reward, terminated, time_out, info

    def render(self):
        """Render the environment."""
        img = self.env.render()
        return img

    def close(self):
        """Close the environment."""
        self.env.close()


def _discover_tasks_modules() -> None:
    """Import all task modules under the `tasks` package recursively.

    This ensures modules containing `@register_task` are imported, so they
    appear in the registry before Gym registration.
    """
    try:
        base_pkg = __package__.split(".")[0]  # "tasks"
        base_dir = Path(__file__).resolve().parent
        for py_file in base_dir.rglob("*.py"):
            if py_file.name in {"__init__.py", Path(__file__).name}:
                continue
            rel = py_file.relative_to(base_dir).with_suffix("")
            dotted = ".".join((base_pkg, *rel.parts))
            try:
                import_module(dotted)
            except Exception:
                # ignore faulty modules during discovery
                pass
    except Exception:
        pass


def _make_entry_point(task_name: str) -> Callable[..., gym.Env]:
    def _factory(**kwargs: Any) -> gym.Env:
        device = kwargs.pop("device", None)
        return GymEnvWrapper(task_name=task_name, device=device, **kwargs)

    return _factory


def register_all_tasks_with_gym(prefix: str = "RoboVerse/") -> None:
    """Register all tasks from registry with Gymnasium.

    Each task name will be exposed as an env id f"{prefix}{task_name}-v0".
    Example usage:

        from tasks.gym_registration import register_all_tasks_with_gym
        register_all_tasks_with_gym()
        env = gym.make("RoboVerse/reach.origin-v0", robots=["franka"], simulator="mujoco", num_envs=1)
    """
    # ensure tasks are discovered
    _discover_tasks_modules()

    for task_name in list_tasks():
        env_id = f"{prefix}{task_name}-v0"
        # avoid duplicate registration in hot-reload/dev
        try:
            register(id=env_id, entry_point=_make_entry_point(task_name))
        except Exception:
            # ignore if already registered
            pass


def register_task_with_gym(task_name: str, env_id: str | None = None) -> str:
    """Register a single task with Gymnasium.

    Returns the env id.
    """
    if env_id is None:
        env_id = f"RoboVerse/{task_name}-v0"
    try:
        register(id=env_id, entry_point=_make_entry_point(task_name))
    except Exception:
        # ignore if already registered
        pass
    return env_id
