"""Vectorized evaluation rollout over ``num_envs`` with per-env episode bookkeeping.

All envs are stepped in parallel. The rollout is
**wave-based**: each wave full-resets *all* ``num_envs`` and runs one episode per slot;
there is no per-slot auto-reset mid-wave (it deliberately relies on ``BaseTaskEnv.step``
not auto-resetting). A terminated env keeps being stepped but its outcome is *latched*
(success on the first ``terminated``); waves repeat until ``episodes`` outcomes are
counted. Action chunking is handled by a per-env :class:`~.chunking.ChunkScheduler` that
re-queries at each env's own horizon and is reset at the start of every wave. Targets
``BaseTaskEnv`` checker tasks; ``RLTaskEnv`` auto-reset semantics are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chunking import ActionChunk, ChunkScheduler, TemporalEnsembler
from .env_adapter import EnvAdapter
from .obs import ActionBatch
from .policy import Policy


@dataclass
class RolloutResult:
    task: str
    episodes: int
    successes: int
    steps_mean: float
    per_episode_success: tuple[bool, ...]


def _bool(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.reshape(-1).to(torch.bool)


class VecEvalRunner:
    """Runs a :class:`Policy` against an :class:`EnvAdapter` for N episodes, vectorized."""

    def __init__(self, adapter: EnvAdapter, policy: Policy, *, chunk: str = "auto", ensemble_k: float = 0.01) -> None:
        self.adapter = adapter
        self.policy = policy
        self.spec = adapter.action_spec
        self.n = adapter.num_envs
        if self.n < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.n}")
        # Episode bookkeeping lives on the SIM device: `terminated`/`timeout` come back from
        # step() on it, and mixing them with CPU-allocated masks raises "Expected all tensors to
        # be on the same device" on every GPU backend (mjx/isaacgym/isaacsim/newton).
        self.device = torch.device(getattr(adapter, "device", None) or "cpu")
        self.H = self.spec.chunk_len
        if chunk == "auto":
            chunk = "scheduler" if self.H > 1 else "none"
        if chunk not in {"none", "scheduler", "ensemble"}:
            raise ValueError(f"chunk must be 'auto'|'none'|'scheduler'|'ensemble', got {chunk!r}")
        if chunk == "none" and self.H > 1:
            raise ValueError(
                f"chunk='none' but the policy declares chunk_len={self.H}: its (B, {self.H}, *) chunk cannot "
                f"be applied as a single (B, *) action. Use chunk='auto'|'scheduler'|'ensemble' to unroll the "
                "chunk, or set chunk_len=1 in describe()."
            )
        self.chunk = chunk
        self.ensemble_k = ensemble_k
        # chunk handler is built in run() so every wave starts clean. On the sim device so chunked
        # policies on a CUDA backend don't hit a cross-device index-assign.
        self._handler = None  # ChunkScheduler | TemporalEnsembler | None

    def _build_handler(self):
        if self.chunk == "none":
            return None
        if self.chunk == "scheduler":
            return ChunkScheduler(action_spec=self.spec, num_envs=self.n, device=self.device)
        return TemporalEnsembler(action_spec=self.spec, num_envs=self.n, k=self.ensemble_k, device=self.device)

    def _policy_step(self, obs, step: int) -> ActionBatch:
        all_ids = obs.env_ids
        if self.chunk == "none":  # direct, chunk_len==1
            return self.policy.act(obs)
        if self.chunk == "scheduler":
            need = self._handler.needs_query()
            if need.numel() > 0:
                self._handler.push(ActionChunk.from_batch(self._chunked_act(obs.index(need))), need)
            return ActionBatch(self.spec, all_ids, self._handler.action_for(all_ids))
        # ensemble: ALOHA/ACT temporal aggregation — query EVERY step, blend overlapping predictions
        self._handler.push(step, ActionChunk.from_batch(self._chunked_act(obs)), all_ids)
        return ActionBatch(self.spec, all_ids, self._handler.action_for(step, all_ids))

    def _chunked_act(self, obs) -> ActionBatch:
        """Query the policy and fail fast if it violates its declared chunk_len (returns a flat
        action when it advertised chunk_len>1 — which would otherwise silently degrade to
        chunk_len=1 / no ensembling)."""
        ab = self.policy.act(obs)
        ab.validate()  # fail fast on a misshaped policy action at the policy->runner boundary
        if self.H > 1 and not ab.is_chunked:
            raise ValueError(
                f"policy declared chunk_len={self.H} but returned a flat action; return "
                f"(B, {self.H}, *) chunk tensors, or set chunk_len=1 in describe()"
            )
        return ab

    def run(self, *, episodes: int, max_steps: int, seed: int | None = None) -> RolloutResult:
        """Wave-based vectorized rollout.

        Each wave resets ALL ``num_envs`` (a full reset — subset resets need per-env demo
        states many tasks don't slice cleanly) and runs one episode per slot in parallel;
        outcomes are latched per env (success = ``terminated`` fired, once). Each wave uses a
        distinct seed (``seed + wave``) so randomizing tasks give different episodes. Waves
        repeat until ``episodes`` are counted; ``episodes`` is a **floor** — every env in the
        final wave is counted (the true simulated count is returned), so the result may exceed
        ``episodes`` rather than discard fully-simulated envs. Targets ``BaseTaskEnv`` checker
        tasks (no auto-reset in ``step``); ``RLTaskEnv`` is rejected upstream in ``evaluate``.
        """
        adapter, n, dev = self.adapter, self.n, self.device
        all_ids = torch.arange(n, device=dev)
        completed = successes = steps_sum = 0
        per_ep: list[bool] = []
        wave = 0
        self._handler = self._build_handler()  # None for chunk='none'
        while completed < episodes:
            # distinct seed per wave so a randomizing task yields different episodes each wave
            # (reusing one seed would collapse all waves to identical rollouts).
            adapter.reset(seed=(seed + wave) if seed is not None else None)
            wave += 1
            self.policy.reset(all_ids)
            if self._handler is not None:
                self._handler.reset()
            ep_len = torch.zeros(n, dtype=torch.long, device=dev)
            succ = torch.zeros(n, dtype=torch.bool, device=dev)
            done = torch.zeros(n, dtype=torch.bool, device=dev)
            for step in range(max_steps):
                if bool(done.all()):
                    break
                obs = adapter.obs_batch(all_ids)
                out = adapter.apply(self._policy_step(obs, step))
                # a backend may report termination on CPU while the sim runs on GPU (or vice
                # versa); normalize onto the runner's device rather than assuming they agree.
                terminated, timeout = _bool(out[2]).to(dev), _bool(out[3]).to(dev)
                active = ~done
                succ = succ | (active & terminated)  # latch success on first termination
                newly = active & (terminated | timeout)
                ep_len = torch.where(newly, torch.full_like(ep_len, step + 1), ep_len)
                done = done | newly
            ep_len = torch.where(done, ep_len, torch.full_like(ep_len, max_steps))  # timed-out envs
            # count ALL n fully-simulated envs this wave (episodes is a floor, not a hard cap) —
            # discarding completed envs just because they overshoot the target would waste real
            # simulation and make which env is dropped index-order arbitrary.
            for d in range(n):
                completed += 1
                s = bool(succ[d])
                successes += int(s)
                steps_sum += int(ep_len[d])
                per_ep.append(s)
        return RolloutResult(
            task=getattr(adapter.env, "task_name", type(adapter.env).__name__),
            episodes=completed,
            successes=successes,
            steps_mean=steps_sum / max(completed, 1),
            per_episode_success=tuple(per_ep),
        )
