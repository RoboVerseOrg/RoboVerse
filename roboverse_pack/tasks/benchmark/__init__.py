"""Benchmark task definitions."""

from __future__ import annotations

from roboverse_pack.benchmark.spec import BenchmarkTaskSpec

from .cube_reach import BENCHMARK_TASK_SPECS

_TASKS_BY_NAME: dict[str, BenchmarkTaskSpec] = {}
for _spec in BENCHMARK_TASK_SPECS:
    _TASKS_BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _TASKS_BY_NAME[_alias] = _spec


def get_benchmark_task_spec(name: str) -> BenchmarkTaskSpec:
    """Look up a benchmark task spec by canonical name or alias."""
    key = str(name).strip().lower()
    if key and not key.startswith("benchmark."):
        prefixed_key = f"benchmark.{key}"
        if prefixed_key in _TASKS_BY_NAME:
            key = prefixed_key
    try:
        return _TASKS_BY_NAME[key]
    except KeyError as exc:
        available = ", ".join(list_benchmark_task_specs())
        raise KeyError(f"unknown benchmark task {name!r}; available: {available}") from exc


def list_benchmark_task_specs() -> list[str]:
    """Return canonical benchmark task names."""
    return sorted({spec.name for spec in BENCHMARK_TASK_SPECS})


__all__ = [
    "BENCHMARK_TASK_SPECS",
    "get_benchmark_task_spec",
    "list_benchmark_task_specs",
]
