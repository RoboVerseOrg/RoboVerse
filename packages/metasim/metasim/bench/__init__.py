"""``metasim.bench`` — the benchmark protocol: instances, policy contract, evaluator, result schema.

A benchmark that can be *run* needs the same four things whichever task, robot or simulator is
involved: fixed initial conditions (:mod:`~metasim.bench.instances`), one policy contract
(:mod:`~metasim.bench.policy`), one scoring protocol (:mod:`~metasim.bench.protocol`) and one
self-describing result (:mod:`~metasim.bench.result`). :func:`evaluate` ties them together;
``python -m metasim bench`` is its command line. On top of that: multi-task suites with splits and
task-macro scoring (:mod:`~metasim.bench.suite`), per-episode instructions with held-out paraphrases
(:mod:`~metasim.bench.prompts`) and policies served from another process (:mod:`~metasim.bench.remote`).
"""

from __future__ import annotations

from .evaluator import build_context, evaluate
from .instances import TaskInstance, demo_budget, hash_state, instance_pool, plan_instances
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
from .prompts import PromptBank, PromptDraw, task_prompt_bank
from .protocol import PROMPT_SPLITS, SUCCESS_RULES, EvalProtocol
from .remote import PolicyServer, RemotePolicy
from .result import SCHEMA, SCHEMA_VERSION, BenchmarkResult, EpisodeResult, summarize, validate_result, wilson_interval
from .suite import BenchmarkSuite, SuiteTask, get_suite, list_suites, register_suite, suite_report, suite_table

__all__ = [
    "BUILTIN_POLICIES",
    "PROMPT_SPLITS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SUCCESS_RULES",
    "BenchmarkResult",
    "BenchmarkSuite",
    "EpisodeResult",
    "EvalProtocol",
    "HoldPolicy",
    "Policy",
    "PolicyContext",
    "PolicyServer",
    "PromptBank",
    "PromptDraw",
    "RandomPolicy",
    "RemotePolicy",
    "ReplayPolicy",
    "SuiteTask",
    "TaskInstance",
    "build_context",
    "checkpoint_sha256",
    "demo_budget",
    "evaluate",
    "get_suite",
    "hash_state",
    "instance_pool",
    "list_suites",
    "load_policy",
    "plan_instances",
    "register_suite",
    "suite_report",
    "suite_table",
    "summarize",
    "task_prompt_bank",
    "validate_result",
    "wilson_interval",
]
