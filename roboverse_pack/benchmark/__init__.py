"""Benchmark task metadata and registry helpers."""

from __future__ import annotations

from roboverse_pack.benchmark.spec import (
    BenchmarkCameraPreset,
    BenchmarkObjectSpec,
    BenchmarkRobotTeleopProfile,
    BenchmarkSceneSpec,
    BenchmarkTaskSpec,
)

__all__ = [
    "BenchmarkCameraPreset",
    "BenchmarkObjectSpec",
    "BenchmarkRobotTeleopProfile",
    "BenchmarkSceneSpec",
    "BenchmarkTaskSpec",
    "get_benchmark_task_spec",
    "list_benchmark_task_specs",
]


def __getattr__(name: str):
    # ``roboverse_pack.tasks.benchmark`` imports ``roboverse_pack.benchmark.spec`` at module scope;
    # importing it eagerly here made ``import roboverse_pack.tasks.benchmark`` (e.g. during task
    # discovery) a circular import. Resolve the two helpers on first access instead.
    if name in ("get_benchmark_task_spec", "list_benchmark_task_specs"):
        from roboverse_pack.tasks import benchmark as _tasks_benchmark

        return getattr(_tasks_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
