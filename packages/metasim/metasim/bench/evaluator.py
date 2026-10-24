"""The one evaluation loop: harness-owned, protocol-scored, provenance-stamped.

``evaluate`` drives any ``BaseTaskEnv`` (auto-resetting or not) with any :class:`Policy` over the
instances a protocol plans, reads the task's per-step ``info["success"]``, and returns a
:class:`BenchmarkResult`. Every evaluation script in the tree should be a thin caller of this
function, so that "success rate" means one thing.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger as log

from metasim.utils.log import warn_once

from .instances import TaskInstance, instance_pool, plan_instances
from .policy import Policy, PolicyContext
from .protocol import EvalProtocol
from .result import BenchmarkResult, EpisodeResult


@dataclasses.dataclass
class _Running:
    """Book-keeping for an episode in flight."""

    episode: int
    instance: TaskInstance
    flags: list[bool] = dataclasses.field(default_factory=list)
    steps: int = 0
    return_sum: float = 0.0


def build_context(env, protocol: EvalProtocol) -> PolicyContext:
    """The :class:`PolicyContext` of ``env``: joint order, limits and defaults from the robot configs."""
    handler = env.handler
    scenario = env.scenario
    robots = list(getattr(scenario, "robots", []) or [])
    joint_names = handler.get_action_joint_names()
    low, high, default = [], [], []
    by_name = {r.name: r for r in robots}
    for robot_name, joints in joint_names.items():
        cfg = by_name.get(robot_name)
        limits = getattr(cfg, "joint_limits", None) or {}
        defaults = getattr(cfg, "default_joint_positions", None) or {}
        for j in joints:
            lo, hi = limits.get(j, (-float("inf"), float("inf")))
            low.append(float(lo))
            high.append(float(hi))
            default.append(float(defaults.get(j, 0.0)))
    device = torch.device(getattr(env, "device", None) or handler.device)
    cap = protocol.max_episode_steps or int(getattr(env, "max_episode_steps", 0) or 0)
    return PolicyContext(
        task=type(env).__name__,
        simulator=str(getattr(scenario, "simulator", "")),
        robots=[r.name for r in robots],
        num_envs=int(env.num_envs),
        device=device,
        seed=protocol.seed,
        max_episode_steps=cap,
        action_joint_names={k: list(v) for k, v in joint_names.items()},
        action_low=torch.tensor(low, dtype=torch.float32, device=device),
        action_high=torch.tensor(high, dtype=torch.float32, device=device),
        default_action=torch.tensor(default, dtype=torch.float32, device=device),
        instruction=getattr(env, "task_desc", None) or None,
    )


def _reset(env, env_ids: list[int], instances: list[TaskInstance], seed: int):
    """``env.reset`` for ``env_ids``, with the instances' states when they have one."""
    states = [inst.state for inst in instances]
    if any(s is None for s in states):
        if any(s is not None for s in states):
            # a mixed batch would need two resets; keep the explicit ones and default the rest
            with_state = [i for i, s in enumerate(states) if s is not None]
            without = [i for i, s in enumerate(states) if s is None]
            _reset(env, [env_ids[i] for i in with_state], [instances[i] for i in with_state], seed)
            return _reset(env, [env_ids[i] for i in without], [instances[i] for i in without], seed)
        states = None
    try:
        return env.reset(states=states, env_ids=env_ids, seed=seed)
    except TypeError as err:
        if "seed" not in str(err):
            raise
        warn_once(("bench.reset_no_seed", type(env)), f"{type(env).__name__}.reset() takes no seed; seeding skipped")
        return env.reset(states=states, env_ids=env_ids)


def _fresh_obs(env):
    """The observation after a manual reset (``reset`` returns only the reset envs' rows)."""
    return env._observation(env.handler.get_states(mode="tensor"))


def _as_bools(value, num_envs: int) -> list[bool]:
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    out = [bool(v) for v in value]
    if len(out) != num_envs:
        raise ValueError(f"expected one value per env ({num_envs}), got {len(out)}")
    return out


def _as_floats(value, num_envs: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    out = [float(v) for v in value]
    if len(out) != num_envs:
        raise ValueError(f"expected one reward per env ({num_envs}), got {len(out)}")
    return out


def _step_success(env, info: Any, terminated: list[bool]) -> list[bool]:
    """``info["success"]`` of the step; the ``success_is_termination`` reading for envs that predate it."""
    if isinstance(info, dict) and "success" in info:
        return _as_bools(info["success"], len(terminated))
    warn_once(
        ("bench.no_success_info", type(env)),
        f"{type(env).__name__}.step() reports no info['success']; scoring "
        + ("`terminated` as success" if getattr(env, "success_is_termination", True) else "every step as not succeeded")
        + " (see BaseTaskEnv.success_is_termination)",
    )
    if getattr(env, "success_is_termination", True):
        return list(terminated)
    return [False] * len(terminated)


def evaluate(
    env,
    policy: Policy,
    protocol: EvalProtocol,
    *,
    policy_info: dict[str, Any] | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
    task_name: str | None = None,
) -> BenchmarkResult:
    """Score ``policy`` on ``env`` under ``protocol``.

    Args:
        env: A launched task env (``BaseTaskEnv`` or ``RLTaskEnv`` subclass).
        policy: Anything satisfying :class:`Policy`.
        protocol: Episode count, scoring rule, seed, instance selection.
        policy_info: Extra fields for the result's ``policy`` record (``checkpoint``, ``checkpoint_sha256``,
            hyper-parameters). ``name`` and ``kind`` are filled from the policy when absent.
        on_episode: Called with every finished :class:`EpisodeResult` (progress bars, streaming logs).
        task_name: The registry name of the task (the class name is used when not given).

    Returns:
        The run, with one row per planned episode. Env slots beyond the remaining plan keep stepping
        (the policy still acts for them) but are not scored.
    """
    started = time.perf_counter()
    num_envs = int(env.num_envs)
    success_defined = bool(getattr(env, "defines_success", True))
    if not success_defined:
        log.warning(
            f"bench: {type(env).__name__} defines no success signal (success_is_termination is False and "
            "_success is not overridden); every episode will score as not succeeded"
        )
    pool = instance_pool(env)
    plan = plan_instances(pool, protocol)
    ctx = build_context(env, protocol)
    cap = protocol.max_episode_steps
    auto_reset = bool(getattr(env, "auto_reset", False))

    queue: deque[tuple[int, TaskInstance]] = deque(enumerate(plan))
    running: dict[int, _Running] = {}
    results: list[EpisodeResult] = []

    def start(env_ids: list[int]) -> list[int]:
        """Hand the next planned instances to ``env_ids``; returns the ids that actually start something."""
        started_ids, instances = [], []
        for env_id in env_ids:
            if not queue:
                break
            episode, inst = queue.popleft()
            running[env_id] = _Running(episode=episode, instance=inst)
            started_ids.append(env_id)
            instances.append(inst)
        if started_ids:
            policy.reset(started_ids, instances, ctx)
        return started_ids

    # first episodes: one explicit reset for every slot, so idle slots also sit in a known state
    first_ids = list(range(num_envs))
    first = start(first_ids)
    instances = [running[i].instance for i in first]
    idle = [i for i in first_ids if i not in running]
    _reset(env, first, instances, protocol.seed)
    if idle:
        _reset(env, idle, [TaskInstance(index=-1, seed=protocol.seed)] * len(idle), protocol.seed)
    obs = _fresh_obs(env)

    while running:
        actions = policy.act(obs, ctx)
        obs, reward, terminated, time_out, info = env.step(actions)
        term = _as_bools(terminated, num_envs)
        tout = _as_bools(time_out, num_envs)
        succ = _step_success(env, info, term)
        rew = _as_floats(reward, num_envs)

        finished: list[int] = []
        for env_id, run in list(running.items()):
            run.steps += 1
            run.return_sum += rew[env_id]
            run.flags.append(succ[env_id])
            capped = cap is not None and run.steps >= cap
            if term[env_id] or tout[env_id] or capped:
                first_hit = next((i + 1 for i, f in enumerate(run.flags) if f), None)
                episode = EpisodeResult(
                    index=run.episode,
                    env_id=env_id,
                    instance_index=run.instance.index,
                    instance_id=run.instance.instance_id,
                    seed=run.instance.seed,
                    success=protocol.episode_success(run.flags),
                    steps=run.steps,
                    terminated=term[env_id],
                    time_out=tout[env_id] or (capped and not term[env_id]),
                    return_sum=run.return_sum,
                    first_success_step=first_hit,
                )
                results.append(episode)
                if on_episode is not None:
                    on_episode(episode)
                del running[env_id]
                finished.append(env_id)

        if finished:
            started_ids = start(finished)
            if not running:
                break  # the plan is exhausted; nothing left to put the finished slots back into
            # an auto-resetting env already reset the envs it marked done, to the task's own initial
            # state; a planned instance with an explicit state, or a cap the env did not see, still
            # needs our reset. A non-auto-resetting env needs it always.
            need_reset = [
                i
                for i in finished
                if not auto_reset
                or (i in running and running[i].instance.state is not None)
                or not (term[i] or tout[i])
            ]
            if need_reset:
                insts = [
                    running[i].instance if i in running else TaskInstance(index=-1, seed=protocol.seed)
                    for i in need_reset
                ]
                seed = min(inst.seed for inst in insts)
                _reset(env, need_reset, insts, seed)
                obs = _fresh_obs(env)
            if started_ids and not need_reset and not auto_reset:
                obs = _fresh_obs(env)

    results.sort(key=lambda e: e.index)
    policy_record = {"name": getattr(policy, "name", type(policy).__name__), "kind": type(policy).__qualname__}
    policy_record.update(policy_info or {})
    provenance = _provenance(env, protocol.seed)
    scenario = env.scenario
    return BenchmarkResult(
        task=task_name or type(env).__name__,
        simulator=str(getattr(scenario, "simulator", "")),
        robots=ctx.robots,
        num_envs=num_envs,
        protocol=protocol.to_dict(),
        policy=policy_record,
        episodes=results,
        provenance=provenance,
        wall_time_s=time.perf_counter() - started,
        notes={
            "instance_pool": len(pool),
            "instruction": ctx.instruction,
            "task_class": type(env).__qualname__,
            "success_defined": success_defined,
        },
    )


def _provenance(env, seed: int) -> dict[str, Any]:
    """``Provenance`` of the handler as a dict; a provenance failure must not lose the run."""
    try:
        from metasim.utils.trajectory import provenance_from_handler

        return dataclasses.asdict(provenance_from_handler(env.handler, seed=seed))
    except Exception as err:  # recorded, not raised: the scores are already in hand
        log.warning(f"bench: provenance not recorded ({type(err).__name__}: {err})")
        return {"error": f"{type(err).__name__}: {err}"}
