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
    box = _declared_action_box(env)
    if box is not None:
        # an RL env declares its own (normalised, possibly reduced) action space: that is the contract the
        # policy must meet, not the robots' joint limits; "hold" is then the middle of the box
        low, high = box
        default = [(lo + hi) / 2.0 for lo, hi in zip(low, high, strict=True)]
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


def _declared_action_box(env) -> tuple[list[float], list[float]] | None:
    """``env.action_space`` as ``(low, high)`` lists when it is a non-empty, finite 1-D Box; else None.

    ``BaseTaskEnv`` declares an empty Box by default (its actions are joint targets in handler order,
    bounded by the joint limits); ``RLTaskEnv`` and the manager-based env declare the normalised box the
    learner acts in, which may have fewer slots than the robot has joints (one actuator of two).
    """
    try:
        space = env.action_space
    except Exception:  # a task whose space needs a live property the stub lacks: fall back to joint limits
        return None
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    shape = getattr(space, "shape", None)
    if low is None or high is None or not shape or len(shape) != 1 or shape[0] == 0:
        return None
    low_l = [float(v) for v in low.reshape(-1)]
    high_l = [float(v) for v in high.reshape(-1)]
    if any(v in (float("inf"), float("-inf")) for v in low_l + high_l):
        return None
    return low_l, high_l


def _reset(env, env_ids: list[int], instances: list[TaskInstance], seed: int):
    """``env.reset`` for ``env_ids``, with the instances' states when they have one.

    ``states`` is only passed when an instance carries one, so an env whose ``reset`` takes no
    ``states`` (the manager-based RL env) still evaluates from its default reset; a ``seed`` the env
    does not accept is dropped with a one-shot warning. Returns what ``env.reset`` returned.
    """
    states = [inst.state for inst in instances]
    if any(s is None for s in states) and any(s is not None for s in states):
        # a mixed batch needs two resets: the explicit ones, then the defaults
        with_state = [i for i, s in enumerate(states) if s is not None]
        without = [i for i, s in enumerate(states) if s is None]
        _reset(env, [env_ids[i] for i in with_state], [instances[i] for i in with_state], seed)
        return _reset(env, [env_ids[i] for i in without], [instances[i] for i in without], seed)
    kwargs = {"env_ids": env_ids, "seed": seed}
    if states[0] is not None:
        kwargs["states"] = states
    try:
        return env.reset(**kwargs)
    except TypeError as err:
        message = str(err)
        if "states" in message and "states" in kwargs:
            raise TypeError(
                f"{type(env).__name__}.reset() takes no `states`, so it cannot start from an explicit task "
                "instance; evaluate it with an empty instance pool (default resets) or add `states` to its reset"
            ) from err
        if "seed" not in message:
            raise
        warn_once(("bench.reset_no_seed", type(env)), f"{type(env).__name__}.reset() takes no seed; seeding skipped")
        kwargs.pop("seed")
        return env.reset(**kwargs)


def _obs_after_reset(env, returned):
    """The full observation after a reset: what ``reset`` returned when it covers every env, else re-read."""
    if getattr(env, "auto_reset", False) and isinstance(returned, tuple) and returned:
        return returned[0]  # RLTaskEnv-style reset reads and returns the whole batch
    return _fresh_obs(env)


def _fresh_obs(env):
    """The observation after a manual reset (``reset`` returns only the reset envs' rows)."""
    return env._observation(env.handler.get_states(mode="tensor"))


def _as_bools(value, num_envs: int) -> list[bool]:
    """``value`` as one bool per env; a scalar (a native env's ``{"success": True}``) is broadcast."""
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (bool, int, float)) or value is None:
        value = [bool(value)] * num_envs
    out = [bool(v) for v in value]
    if len(out) == 1 and num_envs > 1:
        out = out * num_envs
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

    def reset_slots(env_ids: list[int]):
        """Reset each slot on its own, with its episode's seed, so ``EpisodeResult.seed`` is the seed applied.

        A slot with nothing left to run is put back to the task's default state (its seed is the
        protocol's) rather than left mid-episode; it is never scored. One call per slot costs one
        reset per episode, which is what makes "episode ``i`` used ``seed + i``" true.
        """
        returned = None
        for env_id in env_ids:
            inst = running[env_id].instance if env_id in running else TaskInstance(index=-1, seed=protocol.seed)
            returned = _reset(env, [env_id], [inst], inst.seed)
        return _obs_after_reset(env, returned)

    start(list(range(num_envs)))
    obs = reset_slots(list(range(num_envs)))

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
            start(finished)
            if not running:
                break  # the plan is exhausted; nothing left to put the finished slots back into
            # every started episode is reset here with its own seed and state, whether or not the env
            # auto-reset the slot already (RLTaskEnv did, to the task's default state, unseeded by us)
            obs = reset_slots(finished)

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
