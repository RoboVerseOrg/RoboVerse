"""MjlabPassthroughEnv — expose ANY mjlab task through MetaSim's gym.make API.

Wraps mjlab.envs.ManagerBasedRlEnv with the gym.Env interface so users can do:

    import roboverse_pack
    import gymnasium as gym
    env = gym.make("MjlabPassthrough/Mjlab-Velocity-Flat-Unitree-G1", num_envs=4096)

This is NOT a physics-level port (mjlab uses mujoco-warp, we don't swap to
MetaSim handler here). Rather, it's a UI-level integration: all 12+ mjlab
tasks become first-class MetaSim envs without porting obs/action/reward.

For deep physics integration (training in MetaSim's MujocoHandler), see
roboverse_pack.tasks.mjlab.cartpole_train.* — those are full 1:1 ports.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch
from loguru import logger as log


# Defer mjlab import — only triggered when user actually instantiates a passthrough env
def _import_mjlab():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    try:
        from mjlab.tasks.tracking.mdp import MotionCommandCfg
    except ImportError:
        MotionCommandCfg = None  # not all installations have tracking
    return ManagerBasedRlEnv, RslRlVecEnvWrapper, load_env_cfg, load_rl_cfg, MotionCommandCfg


class MjlabPassthroughEnv(gym.Env):
    """Single-env wrapper. Use MjlabPassthroughVecEnv for multi-env.

    Args:
        task_id: mjlab task id, e.g. "Mjlab-Velocity-Flat-Unitree-G1".
        num_envs: number of parallel envs (passed to mjlab's batched env). Default 1.
        play: if True, disables observation corruption + termination + sets
            infinite episode_length (useful for inference). Default False (training mode).
        motion_file: required for tracking tasks; path to LAFAN motion npz.
        device: torch device for mjlab env. Default cuda:0 if available.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: str,
        num_envs: int = 1,
        play: bool = False,
        motion_file: str | None = None,
        device: str | None = None,
        height: int = 480,
        width: int = 640,
    ):
        ManagerBasedRlEnv, _, load_env_cfg, _, MotionCommandCfg = _import_mjlab()
        self.task_id = task_id
        self.num_envs = num_envs
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        env_cfg = load_env_cfg(task_id, play=play)
        if motion_file is not None and MotionCommandCfg is not None:
            if "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg):
                env_cfg.commands["motion"].motion_file = str(motion_file)
        env_cfg.scene.num_envs = num_envs
        if hasattr(env_cfg, "viewer"):
            env_cfg.viewer.height = height
            env_cfg.viewer.width = width

        self._mjlab_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
        self.observation_space = self._mjlab_env.observation_space
        self.action_space = self._mjlab_env.action_space

    # --- gym.Env API ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._mjlab_env.seed(seed)
        obs, info = self._mjlab_env.reset()
        return obs, info

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        obs, rew, term, trunc, info = self._mjlab_env.step(action)
        return obs, rew, term, trunc, info

    def render(self):
        img = self._mjlab_env.render()
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    def close(self):
        self._mjlab_env.close()

    # --- mjlab-specific bridges (for power users / training scripts) ---
    @property
    def mjlab_env(self):
        """Direct access to the underlying mjlab ManagerBasedRlEnv."""
        return self._mjlab_env

    @property
    def max_episode_length(self) -> int:
        return self._mjlab_env.unwrapped.max_episode_length

    def get_observations(self):
        return self._mjlab_env.unwrapped.observation_manager.compute()


# Convenience: register mjlab tasks under MetaSim's gym namespace


def register_mjlab_passthrough_tasks(prefix: str = "MjlabPassthrough/") -> list[str]:
    """Register all mjlab tasks as gym envs accessible via gym.make(prefix + task_id).

    Idempotent — gymnasium silently ignores re-registration.
    Returns the list of registered env ids (or attempted-but-already-present).
    """
    from gymnasium.envs.registration import register, registry

    # Discover mjlab task ids
    try:
        from mjlab.tasks.registry import list_tasks

        task_ids = list_tasks()
    except Exception as e:
        log.warning(f"[mjlab_passthrough] could not list mjlab tasks: {e}")
        return []
    registered = []
    failed = []
    for task_id in task_ids:
        env_id = f"{prefix}{task_id}"
        if env_id in registry:
            registered.append(env_id)
            continue
        try:
            register(
                id=env_id,
                entry_point="roboverse_pack.tasks.mjlab._passthrough:MjlabPassthroughEnv",
                kwargs={"task_id": task_id},
            )
            registered.append(env_id)
        except Exception as e:
            failed.append((env_id, str(e)))
    if failed:
        log.warning(f"[mjlab_passthrough] {len(failed)} registration failures:")
        for env_id, err in failed[:3]:
            log.warning(f"  {env_id}: {err}")
    return registered
