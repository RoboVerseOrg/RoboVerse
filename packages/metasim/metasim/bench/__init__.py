"""``metasim.bench`` — the benchmark protocol: instances, policy contract, evaluator, result schema.

A benchmark that can be *run* needs the same four things whichever task, robot or simulator is
involved: fixed initial conditions (:mod:`~metasim.bench.instances`), one policy contract
(:mod:`~metasim.bench.policy`), one scoring protocol (:mod:`~metasim.bench.protocol`) and one
self-describing result (:mod:`~metasim.bench.result`). :func:`evaluate` ties them together;
``python -m metasim bench`` is its command line.
"""

from __future__ import annotations

from .evaluator import build_context, evaluate
from .instances import TaskInstance, hash_state, instance_pool, plan_instances
from .policy import (
    BUILTIN_POLICIES,
    HoldPolicy,
    Policy,
    PolicyContext,
    RandomPolicy,
    ReplayPolicy,
    checkpoint_sha256,
    load_policy,
)
from .protocol import SUCCESS_RULES, EvalProtocol
from .result import SCHEMA, SCHEMA_VERSION, BenchmarkResult, EpisodeResult, summarize, validate_result, wilson_interval

__all__ = [
    "BUILTIN_POLICIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SUCCESS_RULES",
    "BenchmarkResult",
    "EpisodeResult",
    "EvalProtocol",
    "HoldPolicy",
    "Policy",
    "PolicyContext",
    "RandomPolicy",
    "ReplayPolicy",
    "TaskInstance",
    "build_context",
    "checkpoint_sha256",
    "evaluate",
    "hash_state",
    "instance_pool",
    "load_policy",
    "plan_instances",
    "summarize",
    "validate_result",
    "wilson_interval",
]
