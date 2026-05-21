"""ManiSkill3 passthrough — expose all 137 ManiSkill 3 envs through MetaSim's gym.make API.

Wraps mani_skill 3.x envs (which already use gym.make) under a
'ManiSkill3/<env_id>' namespace, so users can do:

    import roboverse_pack
    import gymnasium as gym
    env = gym.make("ManiSkill3/PickCube-v1", num_envs=64)

This is the same pattern as MjlabPassthrough — namespacing existing
gym envs into MetaSim's task ecosystem.

ManiSkill 3 supports humanoid (UnitreeG1/H1), quadruped (AnymalC), classic
manipulation (PickCube etc.), and 100+ envs total.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _ensure_maniskill_imported():
    """Triggers ManiSkill envs registration."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mani_skill
    import mani_skill.envs  # noqa: F401 — triggers env registry side-effect
    return mani_skill


_ALREADY_REGISTERED = False


def list_maniskill3_envs() -> list[str]:
    """Return all ManiSkill 3-registered env IDs (after triggering import)."""
    _ensure_maniskill_imported()
    import gymnasium as gym
    ids = []
    for spec in gym.envs.registry.values():
        ep = spec.entry_point
        ep_mod = ep if isinstance(ep, str) else getattr(ep, "__module__", "")
        if "mani_skill" in str(ep_mod):
            ids.append(spec.id)
    return sorted(ids)


def register_maniskill3_passthrough(prefix: str = "ManiSkill3/") -> list[str]:
    """Register all ManiSkill 3 envs under a 'ManiSkill3/<env_id>' namespace.

    Idempotent. Returns list of registered (or already-present) env ids.
    """
    from gymnasium.envs.registration import register, registry
    _ensure_maniskill_imported()
    import gymnasium as gym

    # Find all mani_skill envs. ManiSkill 3 wraps with functools.partial(mani_skill.envs.make).
    # Identify by the additional_wrappers field which references mani_skill, or by name pattern.
    import functools
    base_ids = []
    for spec in list(gym.envs.registry.values()):
        is_ms = False
        # 1. Check additional_wrappers (ManiSkill always wraps with MSTimeLimit)
        for w in (getattr(spec, "additional_wrappers", None) or []):
            if "mani_skill" in str(getattr(w, "entry_point", "")):
                is_ms = True
                break
        # 2. Check if entry_point is a functools.partial wrapping mani_skill.envs.make
        if not is_ms and isinstance(spec.entry_point, functools.partial):
            target = spec.entry_point.func
            if "mani_skill" in str(getattr(target, "__module__", "")):
                is_ms = True
        if is_ms:
            base_ids.append(spec.id)

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
                entry_point=f"roboverse_pack.tasks.maniskill._passthrough_v3:_make_maniskill_env",
                kwargs={"base_id": base_id},
            )
            registered.append(ns_id)
        except Exception as e:
            failed.append((ns_id, str(e)))
    if failed:
        print(f"[maniskill_passthrough] {len(failed)} registration failures:")
        for ns_id, err in failed[:3]:
            print(f"  {ns_id}: {err}")
    return registered


def _make_maniskill_env(base_id: str, **kwargs: Any):
    """Factory for namespaced env — just gym.make the underlying base id."""
    _ensure_maniskill_imported()
    import gymnasium as gym
    return gym.make(base_id, **kwargs)
