"""The one evaluation loop: harness-owned, protocol-scored, provenance-stamped.

``evaluate`` drives any ``BaseTaskEnv`` (auto-resetting or not) with any :class:`Policy` over the
instances a protocol plans, reads the task's per-step ``info["success"]``, and returns a
:class:`BenchmarkResult`. Every evaluation script in the tree should be a thin caller of this
function, so that "success rate" means one thing.
"""

from __future__ import annotations

import dataclasses
import inspect
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger as log

from metasim.utils.log import warn_once

from .instances import TaskInstance, demo_budget, instance_pool, plan_instances
from .policy import Policy, PolicyContext
from .prompts import DRAW_REVISION, task_prompt_bank
from .protocol import EvalProtocol
from .result import BenchmarkResult, EpisodeResult


@dataclasses.dataclass
class _Running:
    """Book-keeping for an episode in flight."""

    episode: int
    instance: TaskInstance
    seed: int
    """The seed the env was given by the reset that started this episode (see :func:`evaluate`)."""
    flags: list[bool] = dataclasses.field(default_factory=list)
    steps: int = 0
    return_sum: float = 0.0


def _declared_action_box(env) -> tuple[list[float], list[float]] | None:
    """``env.action_space`` as ``(low, high)`` lists when it is a non-empty, finite 1-D Box; else None.

    ``BaseTaskEnv`` declares an empty Box by default (its actions are joint targets in handler order,
    bounded by the joint limits); the manager-based RL env declares the normalised box the learner
    acts in, which may have fewer slots than the robot has joints (one actuator of two).
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


def build_context(env, protocol: EvalProtocol, *, max_episode_steps: int | None = None) -> PolicyContext:
    """The :class:`PolicyContext` of ``env``: joint order, limits and defaults from the robot configs.

    ``max_episode_steps`` is the cap the evaluator applies when it differs from the protocol's own
    (a demo-derived budget); by default the protocol's cap, else the task's.

    The action bounds are the env's declared ``action_space`` when the env maps actions itself
    (``actions_are_joint_targets`` is False) and declares a finite one; otherwise the joint limits in
    action order, with the robots' default joint positions as the "hold" action.
    """
    handler = env.handler
    scenario = env.scenario
    robots = list(getattr(scenario, "robots", []) or [])
    joint_names = handler.get_action_joint_names()
    low, high, default = [], [], []
    default_joint_positions: dict[str, dict[str, float]] = {}
    by_name = {r.name: r for r in robots}
    for robot_name, joints in joint_names.items():
        cfg = by_name.get(robot_name)
        limits = getattr(cfg, "joint_limits", None) or {}
        defaults = getattr(cfg, "default_joint_positions", None) or {}
        default_joint_positions[robot_name] = {j: float(defaults.get(j, 0.0)) for j in joints}
        for j in joints:
            lo, hi = limits.get(j, (-float("inf"), float("inf")))
            low.append(float(lo))
            high.append(float(hi))
            default.append(float(defaults.get(j, 0.0)))
    device = torch.device(getattr(env, "device", None) or handler.device)
    cap = max_episode_steps or protocol.max_episode_steps or int(getattr(env, "max_episode_steps", 0) or 0)
    box = None if getattr(env, "actions_are_joint_targets", True) else _declared_action_box(env)
    if box is not None:
        low, high = box
        default = [(lo + hi) / 2.0 for lo, hi in zip(low, high, strict=True)]  # "hold" = the middle of the box
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
        default_joint_positions=default_joint_positions,
        # under a train / held-out split the instruction of an episode is its instance's prompt; a canonical
        # text here would let a policy that reads the context run every episode on the training text
        instruction=None
        if protocol.prompt_split != "canonical"
        else (getattr(env, "task_desc", None) or (bank.canonical if (bank := task_prompt_bank(env)) else None)),
        extras={"scenario": scenario},  # for adapters that need the robot configs or the camera set-up
    )


def _reset_kwargs(env) -> set[str]:
    """Which of ``states`` / ``env_ids`` / ``seed`` the env's ``reset`` accepts (``**kwargs`` accepts all)."""
    try:
        params = inspect.signature(env.reset).parameters
    except (TypeError, ValueError):
        return {"states", "env_ids", "seed"}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"states", "env_ids", "seed"}
    return {"states", "env_ids", "seed"} & set(params)


def _reset(env, env_ids: list[int], states: list | None, seed: int):
    """``env.reset`` of ``env_ids``, from ``states`` (a full batch, one row per env slot) or the defaults.

    The full-batch form is the one every backend reads the same way: row ``k`` belongs to env ``k``
    and only the rows of ``env_ids`` are written (``ParallelSimWrapper._set_states`` documents the
    rule; Genesis and Isaac Gym index the list by env id). Returns what ``env.reset`` returned.
    """
    accepted = _reset_kwargs(env)
    kwargs: dict[str, Any] = {}
    if "env_ids" in accepted:
        kwargs["env_ids"] = env_ids
    elif env_ids != list(range(env.num_envs)):
        raise TypeError(f"{type(env).__name__}.reset() takes no `env_ids`; it can only be evaluated with num_envs=1")
    if states is not None:
        if "states" not in accepted:
            raise TypeError(
                f"{type(env).__name__}.reset() takes no `states`, so it cannot start from an explicit task "
                "instance; evaluate it with an empty instance pool (default resets) or add `states` to its reset"
            )
        kwargs["states"] = states
    if "seed" in accepted:
        kwargs["seed"] = seed
    else:
        warn_once(("bench.reset_no_seed", type(env)), f"{type(env).__name__}.reset() takes no seed; seeding skipped")
    return env.reset(**kwargs)


def _fresh_obs(env):
    """The observation rebuilt through the task's ``_observation`` hook from the handler state."""
    return env._observation(env.handler.get_states(mode="tensor"))


def _obs_after_reset(env, returned, covered_all: bool, previous_obs):
    """The full observation after a reset.

    What ``reset`` returned when one call covered every env (a single env, or an auto-resetting env,
    whose ``reset`` reads the whole batch); otherwise the observation rebuilt from the handler state
    through ``_observation``. For an env whose ``step`` returns something else than ``_observation``
    does (a native wrapper returning its own tensor or dict), that rebuild differs in type from the step
    observation: say so once rather than hand the policy the wrong thing silently.
    """
    if covered_all and isinstance(returned, tuple) and returned:
        return returned[0]
    fresh = _fresh_obs(env)
    if previous_obs is not None and type(previous_obs) is not type(fresh):
        warn_once(
            ("bench.obs_type_after_reset", type(env)),
            f"{type(env).__name__}: step() returns {type(previous_obs).__name__} observations but a partial reset "
            f"can only be rebuilt through _observation() ({type(fresh).__name__}); the first observation of an "
            "episode started mid-run has that type. Evaluate with num_envs=1 to avoid it.",
        )
    return fresh


def _step_to_host(terminated, time_out, success, reward, num_envs: int):
    """The four per-step vectors as host lists in one device-to-host copy when they are tensors on one device."""
    tensors = [terminated, time_out, success, reward]
    if (
        all(isinstance(t, torch.Tensor) and t.numel() == num_envs for t in tensors)
        and len({t.device for t in tensors}) == 1
    ):
        packed = torch.stack([t.reshape(-1).to(torch.float64) for t in tensors]).cpu().tolist()
        return [bool(v) for v in packed[0]], [bool(v) for v in packed[1]], [bool(v) for v in packed[2]], packed[3]
    return (
        _as_bools(terminated, num_envs),
        _as_bools(time_out, num_envs),
        _as_bools(success, num_envs),
        _as_floats(reward, num_envs),
    )


def _per_env(value, num_envs: int, cast, what: str) -> list:
    """``value`` as one ``cast`` per env; a scalar (a native env's ``{"success": True}``, reward ``1.0``) is broadcast."""
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (bool, int, float)) or value is None:
        value = [value] * num_envs
    elif hasattr(value, "tolist"):
        value = list(value.reshape(-1).tolist()) if hasattr(value, "reshape") else list(value.tolist())
    out = [cast(v) for v in value]
    if len(out) == 1 and num_envs > 1:
        out = out * num_envs
    if len(out) != num_envs:
        raise ValueError(f"expected one {what} per env ({num_envs}), got {len(out)}")
    return out


def _as_bools(value, num_envs: int) -> list[bool]:
    return _per_env(value, num_envs, lambda v: bool(v), "flag")


def _as_floats(value, num_envs: int) -> list[float]:
    return _per_env(value, num_envs, lambda v: float(v), "reward")


def _step_success(env, info: Any, terminated):
    """``info["success"]`` of the step (unconverted); the ``success_is_termination`` reading for envs that predate it."""
    if isinstance(info, dict) and "success" in info:
        return info["success"]
    warn_once(
        ("bench.no_success_info", type(env)),
        f"{type(env).__name__}.step() reports no info['success']; scoring "
        + ("`terminated` as success" if getattr(env, "success_is_termination", True) else "every step as not succeeded")
        + " (see BaseTaskEnv.success_is_termination)",
    )
    if getattr(env, "success_is_termination", True):
        return terminated
    return False


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

    Seeding: a handler's ``set_seed`` is one RNG for the whole batch, so the episodes started by one
    reset call share a seed, the lowest planned seed among them (``protocol.seed + episode``); with
    ``num_envs=1`` that is exactly ``seed + i`` for episode ``i``. ``EpisodeResult.seed`` records the
    seed the reset that started the episode was given, never a seed that was not applied. Runs are
    reproducible for a given ``num_envs``.

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
    name = task_name or type(env).__name__
    bank = task_prompt_bank(env)
    plan = plan_instances(pool, protocol, prompts=bank, task=name)
    cap, cap_source = protocol.max_episode_steps, "protocol" if protocol.max_episode_steps else "task"
    if protocol.demo_budget_factor is not None:
        cap = demo_budget(pool, factor=protocol.demo_budget_factor, quantile=protocol.demo_budget_quantile)
        cap_source = "demos" if cap is not None else "task"
        if cap is None:
            log.warning(f"bench: {name} has no demo actions to derive a budget from; the task's own cap applies")
        task_cap = int(getattr(env, "max_episode_steps", 0) or 0)
        if cap is not None and task_cap and cap > task_cap:
            warn_once(
                ("bench.budget_over_task_cap", name),
                f"bench: {name}'s demo budget ({cap} steps) exceeds the task's own time-out ({task_cap}); the task "
                f"ends episodes first, so the effective cap is {task_cap}",
            )
            cap, cap_source = task_cap, "task (below the demo budget)"
    ctx = build_context(env, protocol, max_episode_steps=cap)
    auto_reset = bool(getattr(env, "auto_reset", False))

    queue: deque[tuple[int, TaskInstance]] = deque(enumerate(plan))
    running: dict[int, _Running] = {}
    results: list[EpisodeResult] = []
    slot_state: list[Any] = [None] * num_envs  # the explicit state each slot last started from, for full batches

    def start(env_ids: list[int], obs_before):
        """Hand the next planned instances to ``env_ids`` and reset those slots (one batched, seeded call).

        Slots with nothing left to run are put back to the task's default state in the same call (an
        auto-resetting env already did that itself) rather than left mid-episode; they are never scored.
        Returns the observation to act on next, or ``obs_before`` when nothing was reset.
        """
        started: list[tuple[int, int, TaskInstance]] = []
        for env_id in env_ids:
            if not queue:
                break
            episode, inst = queue.popleft()
            started.append((env_id, episode, inst))
        seed = min((inst.seed for _, _, inst in started), default=protocol.seed)
        for env_id, episode, inst in started:
            running[env_id] = _Running(episode=episode, instance=inst, seed=seed)
            slot_state[env_id] = inst.state
        started_ids = [env_id for env_id, _, _ in started]
        if started_ids:
            policy.reset(started_ids, [inst for _, _, inst in started], ctx)
        idle = [i for i in env_ids if i not in started_ids]
        explicit = [i for i in started_ids if slot_state[i] is not None]
        default = [i for i in started_ids if slot_state[i] is None] + ([] if auto_reset else idle)
        calls = 0
        returned = None
        if explicit:
            filler = next(s for s in slot_state if s is not None)
            batch = [s if s is not None else filler for s in slot_state]  # rows outside env_ids are not written
            returned = _reset(env, explicit, batch, seed)
            calls += 1
        if default:
            returned = _reset(env, default, None, seed)
            calls += 1
        if calls == 0:
            return obs_before
        covered_all = calls == 1 and (sorted(explicit or default) == list(range(num_envs)) or auto_reset)
        return _obs_after_reset(env, returned, covered_all, obs_before)

    obs = start(list(range(num_envs)), None)

    while running:
        actions = policy.act(obs, ctx)
        obs, reward, terminated, time_out, info = env.step(actions)
        term, tout, succ, rew = _step_to_host(
            terminated, time_out, _step_success(env, info, terminated), reward, num_envs
        )

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
                    seed=run.seed,
                    success=protocol.episode_success(run.flags),
                    steps=run.steps,
                    terminated=term[env_id],
                    time_out=tout[env_id] or (capped and not term[env_id]),
                    return_sum=run.return_sum,
                    first_success_step=first_hit,
                    prompt=run.instance.prompt,
                    prompt_variant=run.instance.prompt_variant,
                )
                results.append(episode)
                if on_episode is not None:
                    on_episode(episode)
                del running[env_id]
                finished.append(env_id)

        if finished:
            if not queue and not running:
                break  # the plan is exhausted; nothing left to put the finished slots back into
            obs = start(finished, obs)

    results.sort(key=lambda e: e.index)
    policy_record = {"name": getattr(policy, "name", type(policy).__name__), "kind": type(policy).__qualname__}
    policy_record.update(policy_info or {})
    provenance = _provenance(env, protocol.seed)
    scenario = env.scenario
    return BenchmarkResult(
        task=name,
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
            "instruction": ctx.instruction or (bank.canonical if bank else None),
            "task_class": type(env).__qualname__,
            "success_defined": success_defined,
            "episode_cap": cap or (ctx.max_episode_steps or None),
            "episode_cap_source": cap_source,
            **({"prompt_draw_revision": DRAW_REVISION} if protocol.prompt_split != "canonical" else {}),
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
