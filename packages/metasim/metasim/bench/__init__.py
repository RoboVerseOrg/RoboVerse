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

from .access import ACCESS_PROFILES, AccessProfile, get_access_profile, restrict_observation
from .evaluator import build_context, evaluate
from .instances import TaskInstance, demo_budget, hash_state, instance_pool, plan_instances, validate_instances
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
from .record import RecordingPolicy, flatten_initial_state, record_demos
from .remote import ConformanceReport, PolicyServer, RemotePolicy, conformance
from .result import (
    SCHEMA,
    SCHEMA_VERSION,
    BenchmarkResult,
    EpisodeResult,
    matrix,
    summarize,
    validate_result,
    wilson_interval,
)
from .suite import BenchmarkSuite, SuiteTask, get_suite, list_suites, register_suite, suite_report, suite_table

__all__ = [
    "ACCESS_PROFILES",
    "BUILTIN_POLICIES",
    "PROMPT_SPLITS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SUCCESS_RULES",
    "AccessProfile",
    "BenchmarkResult",
    "BenchmarkSuite",
    "ConformanceReport",
    "EpisodeResult",
    "EvalProtocol",
    "HoldPolicy",
    "Policy",
    "PolicyContext",
    "PolicyServer",
    "PromptBank",
    "PromptDraw",
    "RandomPolicy",
    "RecordingPolicy",
    "RemotePolicy",
    "ReplayPolicy",
    "SuiteTask",
    "TaskInstance",
    "build_context",
    "checkpoint_sha256",
    "conformance",
    "demo_budget",
    "evaluate",
    "flatten_initial_state",
    "get_access_profile",
    "get_suite",
    "hash_state",
    "instance_pool",
    "list_suites",
    "load_policy",
    "matrix",
    "plan_instances",
    "record_demos",
    "register_suite",
    "restrict_observation",
    "suite_report",
    "suite_table",
    "summarize",
    "task_prompt_bank",
    "validate_instances",
    "validate_result",
    "wilson_interval",
]
