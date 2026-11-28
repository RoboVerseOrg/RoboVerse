"""Record demonstrations with any policy, in the ``{robot: [demos]}`` layout the task bases and the gate read.

A task needs demos before the replay gate can judge it, and a human-recorded set is the slow path.
:func:`record_demos` drives a task with a policy (the scripted skills, a trained checkpoint, a served
model) through the same evaluator the benchmark uses, keeps the episodes the task reports as successful
(or all of them), and writes one trajectory file: per demo the initial state the episode started from
and the action the policy sent at every step. The result is what ``get_traj`` loads and what
``bench health`` replays, so a recorded set is checked by the gate before it is published::

    python -m metasim bench record --task pick_cube --sim mujoco --episodes 50 --out demos/pick_cube/v2/franka_v2.pkl.gz
    python -m metasim bench health --sim mujoco --task pick_cube --demos demos/pick_cube/v2 --floor --repeat 2
"""

from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .evaluator import evaluate
from .instances import TaskInstance
from .policy import PolicyContext
from .protocol import EvalProtocol
from .result import BenchmarkResult, EpisodeResult


def _to_plain(value: Any) -> Any:
    """Tensors and arrays to lists / floats, recursively, so the file is a plain pickle."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        out = value.tolist()
        return out
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def flatten_initial_state(state: Any) -> dict[str, Any]:
    """A reset state in either layout to the v2 ``{entity: {pos, rot, dof_pos?}}`` layout, as plain Python.

    ``env.reset(states=...)`` accepts the nested ``{"objects": {...}, "robots": {...}}`` form a handler
    produces and the flat form the demo files carry; the demo files use the flat one.
    """
    if not isinstance(state, dict):
        raise TypeError(f"an initial state must be a dict, got {type(state).__name__}")
    entities: dict[str, Any] = {}
    if "objects" in state or "robots" in state:
        for group in ("objects", "robots"):
            for name, ent in (state.get(group) or {}).items():
                entities[name] = ent
    else:
        entities = state
    out: dict[str, Any] = {}
    for name, ent in entities.items():
        if not isinstance(ent, dict):
            continue
        row = {k: _to_plain(ent[k]) for k in ("pos", "rot") if k in ent}
        if ent.get("dof_pos"):
            row["dof_pos"] = _to_plain(ent["dof_pos"])
        out[name] = row
    return out


def _actions_as_dicts(actions: Any, ctx: PolicyContext) -> list[dict]:
    """One ``{robot: {"dof_pos_target": {joint: value}}}`` per env, from a tensor or a per-env dict list."""
    if isinstance(actions, (list, tuple)):
        return [_to_plain(a) for a in actions]
    arr = torch.as_tensor(actions).detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    out = []
    for row in arr.tolist():
        offset, entry = 0, {}
        for robot, joints in ctx.action_joint_names.items():
            entry[robot] = {"dof_pos_target": dict(zip(joints, row[offset : offset + len(joints)], strict=True))}
            offset += len(joints)
        out.append(entry)
    return out


@dataclass
class _Buffer:
    init_state: dict[str, Any] | None
    actions: list[dict] = field(default_factory=list)


class RecordingPolicy:
    """Wraps a policy and keeps, per env slot, the initial state and every action of the episode in flight."""

    def __init__(self, policy, env) -> None:
        self.inner = policy
        self.env = env
        self.name = getattr(policy, "name", type(policy).__name__)
        family = getattr(policy, "family", None)
        if isinstance(family, str):
            self.family = family
        self.current: dict[int, _Buffer] = {}

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Start a buffer per env; the initial state comes from the instance, else from the handler after reset."""
        for env_id, inst in zip(env_ids, instances, strict=True):
            state = getattr(inst, "state", None)
            self.current[env_id] = _Buffer(flatten_initial_state(state) if state is not None else None)
        self.inner.reset(env_ids, instances, ctx)

    def act(self, obs: Any, ctx: PolicyContext) -> Any:
        """Forward to the policy and append its per-env action to each live buffer."""
        actions = self.inner.act(obs, ctx)
        per_env = _actions_as_dicts(actions, ctx)
        for env_id, buf in self.current.items():
            if buf.init_state is None:  # a default reset: read the state the episode actually started from
                nested = _nested_states(self.env)
                buf.init_state = flatten_initial_state(nested[env_id])
            if env_id < len(per_env):
                buf.actions.append(per_env[env_id])
        return actions

    def usage(self):
        """The wrapped policy's usage report, when it has one."""
        usage = getattr(self.inner, "usage", None)
        return usage() if callable(usage) else None

    def pop(self, env_id: int) -> _Buffer | None:
        """Take the finished episode's buffer for ``env_id``."""
        return self.current.pop(env_id, None)


def _nested_states(env) -> list[dict]:
    from metasim.utils.state import state_tensor_to_nested

    return state_tensor_to_nested(env.handler, env.handler.get_states(mode="tensor"))


def record_demos(
    env,
    policy,
    protocol: EvalProtocol,
    *,
    out: str | Path,
    success_only: bool = True,
    task_name: str | None = None,
    on_episode=None,
) -> tuple[BenchmarkResult, dict[str, Any]]:
    """Run ``policy`` on ``env`` under ``protocol`` and write the demos to ``out``.

    Returns the evaluation result and a summary (``{"kept", "dropped", "robot", "path"}``). The file is a
    gzip pickle when ``out`` ends with ``.gz``, a plain pickle otherwise, in the ``{robot: [{"init_state",
    "actions", "states": None}]}`` layout; a JSON sidecar next to it records the policy, protocol and scores.
    """
    robots = [r.name for r in (getattr(env.scenario, "robots", None) or [])]
    if len(robots) != 1:
        raise ValueError(f"record_demos writes single-robot demo files; the scenario has robots {robots}")
    robot = robots[0]
    recorder = RecordingPolicy(policy, env)
    demos: list[dict[str, Any]] = []
    dropped = 0

    def _finished(ep: EpisodeResult) -> None:
        nonlocal dropped
        buf = recorder.pop(ep.env_id)
        if buf is None:
            return
        if ep.success or not success_only:
            demos.append({"init_state": buf.init_state, "actions": buf.actions, "states": None, "success": ep.success})
        else:
            dropped += 1
        if on_episode is not None:
            on_episode(ep)

    result = evaluate(
        env, recorder, protocol, policy_info={"spec": "record"}, on_episode=_finished, task_name=task_name
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the v2 layout keys the file by robot and stores each action as that robot's own dict
    # ({"dof_pos_target": {...}}), not wrapped under the robot name the harness uses for multi-robot batches
    for d in demos:
        d["actions"] = [a.get(robot, a) if isinstance(a, dict) and set(a) == {robot} else a for a in d["actions"]]
    payload = {robot: [{k: v for k, v in d.items() if k != "success"} for d in demos]}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as f:
        pickle.dump(payload, f)
    summary = {
        "task": result.task,
        "robot": robot,
        "simulator": result.simulator,
        "policy": result.policy,
        "protocol": result.protocol,
        "kept": len(demos),
        "dropped": dropped,
        "success_rate": result.summary.get("success_rate"),
        "mean_steps": sum(len(d["actions"]) for d in demos) / len(demos) if demos else None,
        "path": str(path),
    }
    with open(
        path.with_suffix(path.suffix + ".json") if path.suffix != ".gz" else path.with_name(path.name + ".json"), "w"
    ) as f:
        json.dump(summary, f, indent=2, default=str)
    return result, summary
