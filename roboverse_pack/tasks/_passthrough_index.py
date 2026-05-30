"""Unified discovery API for all benchmark passthrough envs.

After importing any roboverse_pack.tasks.* submodule that registers a
passthrough, this function returns a dict {benchmark_name: [env_id, ...]}.

Usage:
    import roboverse_pack.tasks.mjlab
    import roboverse_pack.tasks.maniskill
    import roboverse_pack.tasks.metaworld
    import roboverse_pack.tasks.myosuite
    import roboverse_pack.tasks.gym_robotics
    import roboverse_pack.tasks.dm_control
    from roboverse_pack.tasks._passthrough_index import list_passthrough_envs
    envs = list_passthrough_envs()
    print(envs.keys())  # dict_keys(['MjlabPassthrough', 'ManiSkill3', ...])
    print(sum(len(v) for v in envs.values()))  # total count
"""

from __future__ import annotations

PASSTHROUGH_PREFIXES = [
    "MjlabPassthrough/",
    "ManiSkill3/",
    "MetaWorld/",
    "MyoSuite/",
    "GymRobotics/",
    "DMControl/",
    "Libero/",
]


def list_passthrough_envs() -> dict[str, list[str]]:
    """Return {benchmark_name: [env_id, ...]} for all currently-registered passthrough envs."""
    import gymnasium as gym

    out: dict[str, list[str]] = {}
    for prefix in PASSTHROUGH_PREFIXES:
        bench = prefix.rstrip("/")
        ids = sorted([s for s in gym.envs.registry if s.startswith(prefix)])
        if ids:
            out[bench] = ids
    return out


def print_passthrough_summary() -> None:
    """Print a one-line summary per benchmark."""
    envs = list_passthrough_envs()
    total = sum(len(v) for v in envs.values())
    print(f"# RoboVerse Passthrough - {total} envs across {len(envs)} benchmarks")  # noqa: T201
    print()  # noqa: T201
    for bench, ids in envs.items():
        print(f"  {bench:25s} {len(ids):4d}  example: {ids[0]}")  # noqa: T201
    print()  # noqa: T201
    print(f"  TOTAL                     {total}")  # noqa: T201
