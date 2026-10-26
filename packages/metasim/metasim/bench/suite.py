"""Benchmark suites: named sets of tasks in splits, scored per task and macro-averaged per split.

A single success rate pooled over every episode of a multi-task benchmark is dominated by whichever
tasks contribute the most episodes, and silently skips tasks that crashed. A suite fixes the task
list up front and reports, per split, the **task-macro** success rate (the mean of the per-task
rates), the pooled rate next to it, and the tasks that have no result — a suite score with missing
tasks is reported as incomplete rather than as a number over the tasks that happened to run.

Splits carry the generalisation axis: e.g. ``"core"`` tasks a policy may train on, ``"heldout"`` tasks
(new objects, compositions, embodiments) it is only evaluated on. A suite is content: RoboVerse
registers its suites from ``<root>.suites`` packages (``metasim.toml`` roots, the ``metasim.suites``
entry point group or ``METASIM_SUITE_PACKAGES``), the same discovery the task registry uses.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger as log

from .protocol import EvalProtocol
from .result import BenchmarkResult, wilson_interval


@dataclass(frozen=True)
class SuiteTask:
    """One task of a suite.

    Attributes:
        task: Registry name.
        robots: Robot config names to evaluate with; empty = the task's own.
        protocol: Field overrides on the suite protocol for this task (``{"episodes": 50}``).
    """

    task: str
    robots: tuple[str, ...] = ()
    protocol: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "robots", tuple(self.robots))
        object.__setattr__(self, "protocol", dict(self.protocol))


@dataclass(frozen=True)
class BenchmarkSuite:
    """A fixed task list, in splits, under one protocol.

    Attributes:
        name: Suite id (``bench suite run --suite <name>``).
        splits: ``split -> tasks`` (registry names or :class:`SuiteTask`), in report order.
        protocol: The protocol every task is scored under, before per-task overrides.
        simulator: Backend the suite is defined on; ``None`` = the caller picks.
        description: One paragraph for the docs and the report header.
    """

    name: str
    splits: Mapping[str, tuple[SuiteTask | str, ...]]
    protocol: EvalProtocol = field(default_factory=EvalProtocol)
    simulator: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.splits:
            raise ValueError(f"suite {self.name!r} has no splits")
        normalized: dict[str, tuple[SuiteTask, ...]] = {}
        seen: dict[tuple, str] = {}
        for split, tasks in self.splits.items():
            entries = tuple(t if isinstance(t, SuiteTask) else SuiteTask(task=str(t)) for t in tasks)
            if not entries:
                raise ValueError(f"suite {self.name!r}: split {split!r} lists no tasks")
            for entry in entries:
                key = (entry.task, entry.robots)
                if key in seen:
                    raise ValueError(
                        f"suite {self.name!r}: {entry.task} {list(entry.robots) or ''} is in both "
                        f"{seen[key]!r} and {split!r}; a task belongs to one split"
                    )
                seen[key] = split
                bad = set(entry.protocol) - set(EvalProtocol.__dataclass_fields__)
                if bad:
                    raise ValueError(
                        f"suite {self.name!r}: {entry.task} overrides unknown protocol fields {sorted(bad)}"
                    )
            normalized[str(split)] = entries
        object.__setattr__(self, "splits", normalized)

    def entries(self, split: str | None = None) -> list[tuple[str, SuiteTask]]:
        """``(split, task)`` pairs in suite order, optionally of one split."""
        if split is not None and split not in self.splits:
            raise ValueError(f"suite {self.name!r} has no split {split!r}; it has {list(self.splits)}")
        return [(s, t) for s, tasks in self.splits.items() if split in (None, s) for t in tasks]

    def protocol_for(self, entry: SuiteTask) -> EvalProtocol:
        """The suite protocol with ``entry``'s overrides, named after the suite."""
        base = replace(self.protocol, name=self.name) if self.protocol.name == "default" else self.protocol
        return replace(base, **entry.protocol) if entry.protocol else base


_SUITES: dict[str, BenchmarkSuite] = {}
_DISCOVERED = False


def register_suite(suite: BenchmarkSuite) -> BenchmarkSuite:
    """Make ``suite`` available by name; registering another suite under a taken name is an error."""
    existing = _SUITES.get(suite.name)
    if existing is not None and existing != suite:
        raise ValueError(f"a different suite is already registered as {suite.name!r}")
    _SUITES[suite.name] = suite
    return suite


def _discover() -> None:
    """Import every ``suites`` content package once (they register on import)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from metasim.utils.package_discovery import get_package_candidates

    for package in get_package_candidates("suites"):
        try:
            importlib.import_module(package)
        except ModuleNotFoundError as err:
            if err.name and (package == err.name or package.startswith(err.name + ".")):
                continue  # this root ships no suites
            raise


def list_suites() -> list[str]:
    """Names of every registered suite (content packages are imported on first call)."""
    _discover()
    return sorted(_SUITES)


def get_suite(name: str) -> BenchmarkSuite:
    """The suite registered as ``name``."""
    _discover()
    try:
        return _SUITES[name]
    except KeyError:
        raise ValueError(f"no suite named {name!r}; registered: {sorted(_SUITES)}") from None


def _matches(result: BenchmarkResult, split: str, entry: SuiteTask, suite: BenchmarkSuite) -> bool:
    tag = result.notes.get("suite") if isinstance(result.notes, dict) else None
    if isinstance(tag, dict) and (tag.get("name") != suite.name or tag.get("split") != split):
        return False
    if result.task != entry.task:
        return False
    return not entry.robots or tuple(result.robots) == entry.robots


def suite_report(
    suite: BenchmarkSuite, results: Iterable[BenchmarkResult], *, split: str | None = None
) -> dict[str, Any]:
    """Score ``results`` against ``suite``: per task, per split (task-macro and pooled), overall.

    A result counts for a suite task when its task name (and robots, when the suite pins them)
    match, and — for results written by ``bench suite run`` — its suite and split tags agree. Runs
    of one task are pooled. A split's ``macro_success_rate`` is the mean over its tasks and is
    ``None`` while any task is missing or has no defined success signal (``complete`` is False).
    """
    results = list(results)
    simulators = sorted({r.simulator for r in results})
    if len(simulators) > 1:
        raise ValueError(f"results from several simulators {simulators}; report one backend at a time")
    report: dict[str, Any] = {
        "suite": suite.name,
        "simulator": simulators[0] if simulators else suite.simulator,
        "splits": {},
    }
    for split_name in [split] if split else list(suite.splits):
        rows = []
        for _, entry in suite.entries(split_name):
            runs = [r for r in results if _matches(r, split_name, entry, suite)]
            episodes = [e for r in runs for e in r.episodes]
            defined = bool(runs) and all(r.notes.get("success_defined", True) for r in runs)
            k, n = sum(1 for e in episodes if e.success), len(episodes)
            rows.append({
                "task": entry.task,
                "robots": list(entry.robots) or (list(runs[0].robots) if runs else []),
                "runs": len(runs),
                "episodes": n,
                "successes": k,
                "success_rate": (k / n) if (n and defined) else None,
                "success_rate_ci95": list(wilson_interval(k, n)) if (n and defined) else None,
                "status": "missing" if not runs else ("undefined" if not defined else "scored"),
            })
        scored = [r for r in rows if r["success_rate"] is not None]
        complete = len(scored) == len(rows)
        k = sum(r["successes"] for r in scored)
        n = sum(r["episodes"] for r in scored)
        report["splits"][split_name] = {
            "tasks": rows,
            "complete": complete,
            "missing": [r["task"] for r in rows if r["status"] != "scored"],
            "macro_success_rate": (sum(r["success_rate"] for r in scored) / len(scored)) if complete else None,
            "macro_success_rate_scored": (sum(r["success_rate"] for r in scored) / len(scored)) if scored else None,
            "pooled_success_rate": (k / n) if n else None,
            "episodes": n,
        }
    return report


def suite_table(report: dict[str, Any]) -> str:
    """Markdown rendering of :func:`suite_report`: one table per split, then a summary line each."""

    def pct(v):
        return "-" if v is None else f"{v:.3f}"

    lines = [f"## {report['suite']} ({report['simulator'] or 'any simulator'})", ""]
    for split, block in report["splits"].items():
        lines += [
            f"### {split}",
            "",
            "| task | robots | episodes | success rate | 95% CI | status |",
            "|---|---|---|---|---|---|",
        ]
        for r in block["tasks"]:
            ci = (
                "-"
                if r["success_rate_ci95"] is None
                else f"[{r['success_rate_ci95'][0]:.3f}, {r['success_rate_ci95'][1]:.3f}]"
            )
            lines.append(
                f"| {r['task']} | {', '.join(r['robots'])} | {r['episodes']} | {pct(r['success_rate'])} | {ci} | {r['status']} |"
            )
        if block["complete"]:
            head = f"task-macro {pct(block['macro_success_rate'])}"
        else:
            head = (
                f"INCOMPLETE ({len(block['missing'])} task(s) without a score: {', '.join(block['missing'])}); "
                f"macro over scored tasks {pct(block['macro_success_rate_scored'])}"
            )
        lines += [
            "",
            f"{split}: {head}, pooled {pct(block['pooled_success_rate'])} over {block['episodes']} episodes",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def _reset_registry_for_tests() -> None:
    global _DISCOVERED
    _SUITES.clear()
    _DISCOVERED = False
    log.debug("bench suite registry cleared")
