"""MyoSuite passthrough — expose 204 musculoskeletal envs through gym.make."""

from __future__ import annotations

from typing import Any


def _ensure_myosuite_imported():
    import myosuite  # registers envs as side effect
    return myosuite


def register_myosuite_passthrough(prefix: str = "MyoSuite/") -> list[str]:
    """Register all myosuite envs under 'MyoSuite/<env_id>' namespace.

    Returns list of registered (or already-present) env ids.
    """
    from gymnasium.envs.registration import register, registry
    _ensure_myosuite_imported()
    import gymnasium as gym

    base_ids = [s for s in gym.envs.registry if s.startswith("myo")]
    registered = []
    failed = []
    for base_id in base_ids:
        ns_id = f"{prefix}{base_id}"
        if ns_id in registry:
            registered.append(ns_id)
            continue
        try:
            register(
                id=ns_id,
                entry_point="roboverse_pack.tasks.myosuite._passthrough:_make_myo_env",
                kwargs={"base_id": base_id},
            )
            registered.append(ns_id)
        except Exception as e:
            failed.append((ns_id, str(e)))
    if failed:
        print(f"[myosuite_passthrough] {len(failed)} failures")
    return registered


def _make_myo_env(base_id: str, **kwargs: Any):
    _ensure_myosuite_imported()
    import gymnasium as gym
    return gym.make(base_id, **kwargs)
