"""Task instances: the fixed initial conditions an evaluation runs on.

A benchmark number is only comparable when every submission starts from the same conditions. An
instance is one initial state (plus the demo actions that were recorded from it, when there are
any) with a content-addressed id, so a result can say *which* conditions it was scored on and a
second machine can draw the same ones. The pool comes from the task itself: its demo trajectory
file when it has one (``traj_filepath``), otherwise the task's own per-env initial states; a task
with neither is evaluated from its default reset, and the instance then carries only its seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .protocol import EvalProtocol


def _canonical(value: Any) -> Any:
    """Nested state → JSON-able structure with a fixed key order and float32-rounded numbers."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32).round(6).tolist()
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in value):
            return np.asarray(value, dtype=np.float32).round(6).tolist()  # a plain list of numbers == a tensor
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.number, int, float)):
        return float(np.float32(value).round(6))
    return repr(value)


def hash_state(state: Any) -> str:
    """SHA-256 of a nested initial state (dict of tensors / floats), stable across devices and dtypes."""
    payload = json.dumps(_canonical(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TaskInstance:
    """One initial condition to evaluate from."""

    index: int
    """Index into the task's instance pool; with an empty pool, the episode number."""
    seed: int
    state: Any | None = None
    """A nested initial state accepted by ``env.reset(states=[state], env_ids=[i])``; ``None`` = default reset."""
    actions: list | None = None
    """Demo actions recorded from ``state`` (the replay baseline consumes them); ``None`` when unknown."""
    state_hash: str = ""

    @property
    def instance_id(self) -> str:
        """``index:seed:hash12`` — the id written into results."""
        return f"{self.index}:{self.seed}:{self.state_hash[:12] or 'default'}"


def instance_pool(env) -> list[TaskInstance]:
    """Every distinct initial condition the task ships, in file order, with their demo actions.

    Reads the demo file (``get_traj``) rather than ``env._initial_states``, because the task bases tile
    that list to ``num_envs`` and the distinct instances are lost in the process; a task without a demo
    file falls back to ``_initial_states`` de-duplicated by content, so the pool does not change with
    ``num_envs``. The pool is cached on the env (``_bench_instance_pool``): the demo file is read once
    per env, however many times it is evaluated (the task's own ``__init__`` read it once more already).
    """
    cached = getattr(env, "_bench_instance_pool", None)
    if cached is not None:
        return list(cached)
    pool = _build_instance_pool(env)
    try:
        env._bench_instance_pool = tuple(pool)
    except AttributeError:
        pass  # an env that forbids new attributes is simply not cached
    return pool


def _build_instance_pool(env) -> list[TaskInstance]:
    traj = getattr(env, "traj_filepath", None)
    scenario = getattr(env, "scenario", None)
    robots = list(getattr(scenario, "robots", []) or [])
    if traj and robots:
        from metasim.utils.demo_util import get_traj

        robot = robots[0] if len(robots) == 1 else robots
        init_states, all_actions, _ = get_traj(traj, robot, getattr(env, "handler", None))
        actions = list(all_actions) + [None] * len(init_states)
        return [
            TaskInstance(index=i, seed=0, state=s, actions=list(a) if a is not None else None, state_hash=hash_state(s))
            for i, (s, a) in enumerate(zip(init_states, actions, strict=False))
        ]
    states = getattr(env, "_initial_states", None)
    if isinstance(states, list) and states:
        pool, seen = [], set()
        for s in states:
            h = hash_state(s)
            if h in seen:
                continue
            seen.add(h)
            pool.append(TaskInstance(index=len(pool), seed=0, state=s, state_hash=h))
        return pool
    return []


def plan_instances(pool: list[TaskInstance], protocol: EvalProtocol) -> list[TaskInstance]:
    """The instances of one run, in episode order, each with its episode seed.

    Explicit ``protocol.instances`` are taken as given (and must exist in the pool); otherwise the
    pool is cycled from index 0 so ``episodes <= len(pool)`` sees every instance once. An empty pool
    yields seed-only instances.
    """
    planned: list[TaskInstance] = []
    if protocol.instances is not None:
        if not pool:
            raise ValueError("protocol names explicit instances but the task has no instance pool")
        bad = [i for i in protocol.instances if i >= len(pool)]
        if bad:
            raise ValueError(f"instance indices {bad} are outside the pool of {len(pool)}")
        chosen = [pool[i] for i in protocol.instances]
    elif pool:
        chosen = [pool[i % len(pool)] for i in range(protocol.episodes)]
    else:
        chosen = [TaskInstance(index=i, seed=0) for i in range(protocol.episodes)]
    for episode, inst in enumerate(chosen):
        planned.append(
            TaskInstance(
                index=inst.index,
                seed=protocol.seed + episode,
                state=inst.state,
                actions=inst.actions,
                state_hash=inst.state_hash,
            )
        )
    return planned
