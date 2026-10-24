"""The policy side of an evaluation: a small protocol, the context it gets, and three baselines.

The harness owns the loop (reset, step, scoring, result); a policy only maps observations to
actions. That keeps every policy family (RL checkpoints, imitation policies, VLAs behind a server)
on one contract and keeps the scoring out of the policy's hands.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from metasim.types import CompatActionInput

from .instances import TaskInstance


@dataclass
class PolicyContext:
    """What a policy may know about the task it is being evaluated on (fixed for the run)."""

    task: str
    simulator: str
    robots: list[str]
    num_envs: int
    device: torch.device
    seed: int
    max_episode_steps: int
    action_joint_names: dict[str, list[str]]
    """``robot -> joint names`` in the handler's action order (``handler.get_action_joint_names()``)."""
    action_low: torch.Tensor
    """Lower bound of each action slot, shape ``(action_dim,)``: the env's declared ``action_space`` when it
    has a finite, non-empty one (RL envs, normalised), else the joint limits in action order."""
    action_high: torch.Tensor
    default_action: torch.Tensor
    """The "do nothing" action, shape ``(action_dim,)``: the robots' default joint positions in action order,
    or the middle of the declared action box."""
    default_joint_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    """``robot -> {joint: default position}`` for every action joint, whatever the action bounds describe."""
    instruction: str | None = None
    """The task's language instruction (``task_desc``) when it has one."""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def action_dim(self) -> int:
        """Total number of action slots across the robots."""
        return int(self.action_low.shape[0])


@runtime_checkable
class Policy(Protocol):
    """What the evaluator needs from a policy.

    ``reset`` is called before the first step of every episode with the env slots that start one
    and their instances (a replay policy reads the demo actions from them, a VLA reads the
    instruction from the context). ``act`` returns an action for *all* ``num_envs`` slots — a tensor
    ``(num_envs, action_dim)`` in the handler's action order, or a per-env list of
    ``{robot: {"dof_pos_target": {joint: value}}}`` dicts — whatever ``env.step`` accepts.
    """

    name: str

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Start new episodes in ``env_ids``."""

    def act(self, obs: Any, ctx: PolicyContext) -> CompatActionInput:
        """Actions for every env slot given the env's observation."""


class HoldPolicy:
    """Commands the robots' default joint positions every step: the floor every task must beat."""

    name = "hold"

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Nothing to prepare."""

    def act(self, obs: Any, ctx: PolicyContext) -> torch.Tensor:
        """The default joint positions, repeated for every env."""
        return ctx.default_action.unsqueeze(0).repeat(ctx.num_envs, 1)


class RandomPolicy:
    """Uniform joint targets inside the limits, seeded from the protocol so a run is reproducible."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._generator: torch.Generator | None = None

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Seed the generator once, from the explicit seed or the protocol's."""
        if self._generator is None:
            self._generator = torch.Generator(device="cpu").manual_seed(ctx.seed if self._seed is None else self._seed)

    def act(self, obs: Any, ctx: PolicyContext) -> torch.Tensor:
        """Independent uniform draws per env and joint."""
        if self._generator is None:
            raise RuntimeError("RandomPolicy.act called before reset")
        low, high = ctx.action_low.cpu(), ctx.action_high.cpu()
        if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
            bad = [i for i in range(ctx.action_dim) if not (torch.isfinite(low[i]) and torch.isfinite(high[i]))]
            raise ValueError(
                f"RandomPolicy needs finite action bounds; slots {bad} are unbounded (a robot without joint_limits, "
                "or an env whose action_space is infinite). Give the robot limits or use another baseline."
            )
        u = torch.rand((ctx.num_envs, ctx.action_dim), generator=self._generator)
        return (low + u * (high - low)).to(ctx.device)


class ReplayPolicy:
    """Replays the demo actions recorded for each instance; holds the last action once they run out.

    This is the task's own ceiling: a healthy task replays its demos to a high success rate on the
    backend they were recorded on, and the same run on another backend measures how far the demos
    transfer. A slot with no instance (idle env) or an instance without actions gets the hold action.
    """

    name = "replay"

    def __init__(self) -> None:
        self._actions: dict[int, list] = {}
        self._step: dict[int, int] = {}

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Load the demo actions of the instances starting in ``env_ids``."""
        for env_id, inst in zip(env_ids, instances, strict=True):
            self._actions[env_id] = list(inst.actions) if inst.actions else []
            self._step[env_id] = 0

    def act(self, obs: Any, ctx: PolicyContext) -> list:
        """One demo action per env slot, as the dict-action batch ``env.step`` accepts."""
        batch = []
        hold = None
        for env_id in range(ctx.num_envs):
            actions = self._actions.get(env_id, [])
            if actions:
                t = min(self._step.get(env_id, 0), len(actions) - 1)
                self._step[env_id] = self._step.get(env_id, 0) + 1
                batch.append(actions[t])
            else:
                if hold is None:
                    hold = _hold_dict_action(ctx)
                batch.append(hold)
        return batch


def _hold_dict_action(ctx: PolicyContext) -> dict:
    """The robots' default joint positions as a ``{robot: {"dof_pos_target": {...}}}`` action."""
    if ctx.default_joint_positions:
        return {robot: {"dof_pos_target": dict(q)} for robot, q in ctx.default_joint_positions.items()}
    out, offset = {}, 0
    values = ctx.default_action.tolist()
    for robot, joints in ctx.action_joint_names.items():
        out[robot] = {"dof_pos_target": dict(zip(joints, values[offset : offset + len(joints)], strict=True))}
        offset += len(joints)
    return out


BUILTIN_POLICIES: dict[str, type] = {"hold": HoldPolicy, "random": RandomPolicy, "replay": ReplayPolicy}


def load_policy(spec: str, **kwargs: Any) -> Policy:
    """Build a policy from ``"hold"`` / ``"random"`` / ``"replay"`` or an import path ``"pkg.module:attr"``.

    ``attr`` is called with ``kwargs`` when it is a class or a factory; an object that already
    satisfies :class:`Policy` is returned as is (``kwargs`` must then be empty).
    """
    if spec in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[spec](**kwargs)
    if ":" not in spec:
        raise ValueError(
            f"unknown policy {spec!r}: use one of {sorted(BUILTIN_POLICIES)} or an import path 'pkg.module:attr'"
        )
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    try:
        target = getattr(module, attr)
    except AttributeError as err:
        raise ValueError(f"{module_name} has no attribute {attr!r}") from err
    policy = target(**kwargs) if callable(target) else target
    if kwargs and not callable(target):
        raise ValueError(f"{spec} is an instance; it takes no arguments")
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} does not satisfy metasim.bench.Policy (needs name, reset(), act())")
    return policy


def checkpoint_sha256(path: str | Path | None) -> str | None:
    """Hash of a checkpoint file, so a result says exactly which weights produced it."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
