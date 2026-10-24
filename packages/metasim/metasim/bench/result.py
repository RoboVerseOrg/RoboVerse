"""The result of an evaluation run: a versioned, self-describing JSON record.

One file per run, readable without MetaSim: the task, the robots, the simulator, the protocol that
produced the number, the policy (with its checkpoint hash when it has one), one row per episode
and the provenance of the simulation (backend versions, time base, asset hashes, git commit — the
same :class:`~metasim.utils.trajectory.Provenance` an episode record carries). Two results are
comparable when their ``task``, ``robots``, ``simulator`` and ``protocol`` agree; :func:`summarize`
groups them that way.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "roboverse.bench.result"
SCHEMA_VERSION = 1

_REQUIRED = ("schema", "schema_version", "task", "simulator", "robots", "num_envs", "protocol", "policy", "episodes")


@dataclass
class EpisodeResult:
    """One scored episode."""

    index: int
    """Position in the run (``0 .. episodes-1``)."""
    env_id: int
    """Parallel env slot the episode ran in."""
    instance_index: int
    """Index into the task's instance pool (see ``metasim.bench.instances``)."""
    instance_id: str
    """Content-addressed id of the instance (index, seed and a hash of the initial state)."""
    seed: int
    success: bool
    """The protocol's verdict (``EvalProtocol.episode_success``)."""
    steps: int
    terminated: bool
    time_out: bool
    return_sum: float
    first_success_step: int | None = None
    """First step (1-based) at which the task's success predicate held; ``None`` if never."""


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a success rate (well-behaved at 0/N and N/N, unlike the normal one)."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class BenchmarkResult:
    """A complete evaluation run."""

    task: str
    simulator: str
    robots: list[str]
    num_envs: int
    protocol: dict[str, Any]
    policy: dict[str, Any]
    """``{"name": ..., "kind": ..., "checkpoint": ..., "checkpoint_sha256": ..., ...}`` — what was evaluated."""
    episodes: list[EpisodeResult]
    provenance: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate of ``episodes``: count, success rate with a 95% Wilson interval, mean length and return."""
        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.success)
        low, high = wilson_interval(successes, n)
        return {
            "episodes": n,
            "successes": successes,
            "success_rate": successes / n if n else 0.0,
            "success_rate_ci95": [low, high],
            "mean_steps": sum(e.steps for e in self.episodes) / n if n else 0.0,
            "mean_return": sum(e.return_sum for e in self.episodes) / n if n else 0.0,
            "time_outs": sum(1 for e in self.episodes if e.time_out),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; the computed ``summary`` is included for readers without this module."""
        data = asdict(self)
        data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Build from a dict produced by :meth:`to_dict`; validates first."""
        validate_result(data)
        kwargs = {k: v for k, v in data.items() if k != "summary"}
        kwargs["episodes"] = [EpisodeResult(**e) for e in kwargs["episodes"]]
        return cls(**kwargs)

    def save(self, path: str | Path) -> Path:
        """Write the result as JSON (parent directories are created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False, default=_json_default) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Read a result written by :meth:`save`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _json_default(value):
    """Provenance may carry numpy scalars or Paths; everything else is a bug worth seeing."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def validate_result(data: dict[str, Any]) -> None:
    """Raise ``ValueError`` naming the first thing wrong with a result dict."""
    if not isinstance(data, dict):
        raise ValueError(f"a result is a JSON object, got {type(data).__name__}")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"result is missing {missing}")
    if data["schema"] != SCHEMA:
        raise ValueError(f"schema is {data['schema']!r}, expected {SCHEMA!r}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version is {data['schema_version']!r}, this reader understands {SCHEMA_VERSION}")
    if not isinstance(data["episodes"], list):
        raise ValueError("episodes must be a list")
    fields = set(EpisodeResult.__dataclass_fields__)
    for i, episode in enumerate(data["episodes"]):
        if not isinstance(episode, dict):
            raise ValueError(f"episodes[{i}] is not an object")
        unknown = set(episode) - fields
        absent = {f for f in fields if f != "first_success_step"} - set(episode)
        if unknown or absent:
            raise ValueError(f"episodes[{i}] has unknown fields {sorted(unknown)} / missing {sorted(absent)}")
    policy = data["policy"]
    if not isinstance(policy, dict) or "name" not in policy:
        raise ValueError('policy must be an object with at least a "name"')
    if not isinstance(data["protocol"], dict) or "success_rule" not in data["protocol"]:
        raise ValueError('protocol must be an object with at least a "success_rule"')


def result_key(result: BenchmarkResult) -> tuple:
    """What makes two results comparable: task, robots, simulator and the scoring protocol."""
    protocol = result.protocol
    instances = protocol.get("instances")
    return (
        result.task,
        tuple(result.robots),
        result.simulator,
        protocol.get("success_rule"),
        protocol.get("hold_steps"),
        protocol.get("max_episode_steps"),
        tuple(instances) if instances is not None else None,  # hand-picked instances are their own key
    )


def summarize(results: list[BenchmarkResult]) -> str:
    """A Markdown table, one row per (comparable key, policy), pooled over the runs that share both.

    Runs of the same policy on the same task/robot/simulator/protocol (different seeds, different
    machines) are pooled into one success rate with its Wilson interval, which is what a leaderboard
    row is.
    """
    groups: dict[tuple, list[BenchmarkResult]] = {}
    for r in results:
        groups.setdefault(result_key(r) + (r.policy.get("name"),), []).append(r)
    lines = [
        "| task | robots | simulator | rule | policy | runs | episodes | success rate | 95% CI | mean steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        runs = groups[key]
        task, robots, simulator, rule, hold, cap, instances, policy = key
        episodes = [e for r in runs for e in r.episodes]
        n = len(episodes)
        k = sum(1 for e in episodes if e.success)
        low, high = wilson_interval(k, n)
        rule_label = f"{rule}" + (f" x{hold}" if rule == "held" else "") + (f", cap {cap}" if cap else "")
        if instances is not None:
            rule_label += f", instances {list(instances)}"
        mean_steps = sum(e.steps for e in episodes) / n if n else 0.0
        lines.append(
            f"| {task} | {', '.join(robots)} | {simulator} | {rule_label} | {policy} | {len(runs)} | {n} | "
            f"{k / n:.3f} | [{low:.3f}, {high:.3f}] | {mean_steps:.1f} |"
            if n
            else f"| {task} | {', '.join(robots)} | {simulator} | {rule_label} | {policy} | {len(runs)} | 0 | - | - | - |"
        )
    return "\n".join(lines)
