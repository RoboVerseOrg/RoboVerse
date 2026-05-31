"""Expose native robosuite environments through the gym registry.

Registers ``Robosuite/<EnvName>`` ids that ``gym.make`` into the unmodified
robosuite env wrapped in robosuite's own ``GymWrapper``. This is the
lowest-friction integration tier — the env *is* robosuite, so trajectories and
trained-policy performance are identical by construction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

# robosuite env name -> default robot for the passthrough wrapper
_PASSTHROUGH_ROBOTS = {
    "Lift": "Panda",
    "Stack": "Panda",
    "Door": "Panda",
    "PickPlaceCan": "Panda",
    "NutAssemblySquare": "Panda",
}


def _make_robosuite_gym_env(env_name: str, robot: str = "Panda", controller: str = "BASIC", **kwargs: Any):
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.wrappers import GymWrapper

    cfg = load_composite_controller_config(controller=controller, robot=robot)
    env = robosuite.make(
        env_name,
        robots=robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        **kwargs,
    )
    return GymWrapper(env)


def register_robosuite_passthrough(prefix: str = "Robosuite/") -> list[str]:
    """Register all known robosuite envs under ``<prefix><EnvName>``. Idempotent."""
    from gymnasium.envs.registration import register, registry

    registered: list[str] = []
    for env_name, robot in _PASSTHROUGH_ROBOTS.items():
        gid = f"{prefix}{env_name}"
        if gid in registry:
            registered.append(gid)
            continue
        register(
            id=gid,
            entry_point="roboverse_pack.tasks.robosuite._passthrough:_make_robosuite_gym_env",
            kwargs={"env_name": env_name, "robot": robot},
        )
        registered.append(gid)
    return registered
