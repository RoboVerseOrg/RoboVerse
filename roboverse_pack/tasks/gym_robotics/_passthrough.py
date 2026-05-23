"""gymnasium-robotics passthrough — Fetch + Adroit + Shadow Hand + AntMaze + Franka + ..."""

from __future__ import annotations

from typing import Any


def _ensure_imported():
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gymnasium_robotics


def register_gymnasium_robotics_passthrough(prefix: str = "GymRobotics/") -> list[str]:
    """Register all gymnasium-robotics envs under GymRobotics/<env_id> namespace."""
    from gymnasium.envs.registration import register, registry

    _ensure_imported()
    import gymnasium as gym

    # Filter strictly by entry_point being a gymnasium_robotics module/class
    base_ids = []
    for spec in gym.envs.registry.values():
        ep = spec.entry_point
        ep_str = ep if isinstance(ep, str) else f"{getattr(ep, '__module__', '')}.{getattr(ep, '__name__', '')}"
        if "gymnasium_robotics" in ep_str:
            base_ids.append(spec.id)

    registered = []
    for base_id in base_ids:
        ns_id = f"{prefix}{base_id}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.gym_robotics._passthrough:_make_env",
                kwargs={"base_id": base_id},
            )
            registered.append(ns_id)
        except Exception:
            pass
    return registered


def _make_env(base_id: str, **kwargs: Any):
    _ensure_imported()
    import gymnasium as gym

    return gym.make(base_id, **kwargs)
