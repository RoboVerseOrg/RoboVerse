"""Regenerate robots.md from the robot configs and the robot result files (no hand-written rows).

    python docs/source/dataset_benchmark/dataset/generate_robot_docs.py           # rewrite the page
    python docs/source/dataset_benchmark/dataset/generate_robot_docs.py --check   # exit 1 if the page is stale (CI)

Three generated tables: the configs in ``roboverse_pack.robots`` (class, name, joints, asset formats); the latest
``robot check --all`` sweep per simulator from ``results/robots/<date>-<sim>-check/robots.json``; and the latest
``robot parity`` runs from ``results/robots/<date>-parity-<a>-<b>/<robot>.json``. A number on this page is always
traceable to one of those files.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "metasim"))

import roboverse_pack.robots as robots_pkg
from metasim.scenario.robot import RobotCfg

FIELDS = (("mjcf_path", "MuJoCo"), ("usd_path", "Isaac Sim"), ("urdf_path", "URDF"), ("mjx_mjcf_path", "MJX"))
CHECKS = ("config", "assets", "kinematics", "limits", "actuators", "hold")
RESULTS = ROOT / "results" / "robots"
_CHECK_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)-check$")
_PARITY_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})-parity-(.+)$")


def config_rows() -> list[tuple[str, str, str, str, str]]:
    """``(module, class, joints, asset formats, name)`` for every robot config the pack defines."""
    rows = []
    for mod in pkgutil.iter_modules(robots_pkg.__path__):
        if mod.name.startswith("_") or mod.name in ("robot_template",):
            continue
        try:
            module = importlib.import_module(f"roboverse_pack.robots.{mod.name}")
        except Exception as err:  # a robot that needs an optional library is listed with the error, not hidden
            rows.append((mod.name, "", "", f"import failed: {type(err).__name__}", ""))
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, RobotCfg) or cls is RobotCfg or cls.__module__ != module.__name__:
                continue
            try:
                cfg = cls()
            except Exception as err:
                rows.append((mod.name, cls.__name__, "", f"construct failed: {type(err).__name__}", ""))
                continue
            assets = [
                label for field, label in FIELDS if isinstance(getattr(cfg, field, None), str) and getattr(cfg, field)
            ]
            rows.append((
                mod.name,
                cls.__name__,
                str(getattr(cfg, "num_joints", "") or ""),
                ", ".join(assets),
                getattr(cfg, "name", ""),
            ))
    rows.sort(key=lambda r: r[0])
    return rows


def latest_dirs(pattern: re.Pattern, results: Path = RESULTS) -> dict[str, tuple[str, Path]]:
    """``{key: (date, dir)}`` for the newest directory per key (the simulator, or the engine pair)."""
    out: dict[str, tuple[str, Path]] = {}
    if not results.is_dir():
        return out
    for d in sorted(results.iterdir()):
        m = pattern.match(d.name)
        if not m or not d.is_dir():
            continue
        date, key = m.group(1), m.group(2)
        if key not in out or out[key][0] < date:
            out[key] = (date, d)
    return out


def render_check(sim: str, date: str, data: dict) -> list[str]:
    """The ``robot check --all`` sweep as a table plus the named failures."""
    robots = data.get("robots", [])
    n_na = sum(1 for r in robots if not r.get("applicable", True))
    n_ok = sum(1 for r in robots if r.get("ok") and r.get("applicable", True))
    lines = [
        f"### Self-check on {sim} ({date})",
        "",
        f"`python -m metasim robot check --all --sim {sim}`: {len(robots)} robots, {n_ok} pass, "
        f"{len(robots) - n_ok - n_na} fail, {n_na} n/a (no asset declared for this backend). "
        f"Source: `results/robots/{date}-{sim}-check/robots.json`.",
        "",
        "| robot | " + " | ".join(CHECKS) + " | verdict |",
        "|---|" + "---|" * (len(CHECKS) + 1),
    ]

    def cell(row):
        if not row.get("applicable", True):
            return "n/a"
        return "ok" if row["ok"] else "FAIL"

    for r in robots:
        by = {row["check"]: row for row in r.get("rows", [])}
        cells = ["-" if c not in by else cell(by[c]) for c in CHECKS]
        verdict = r.get("verdict") or ("pass" if r.get("ok") else "FAIL")
        lines.append(f"| `{r['robot']}` | " + " | ".join(cells) + f" | {verdict} |")
    failures = [
        (r["robot"], row) for r in robots for row in r.get("rows", []) if row.get("applicable", True) and not row["ok"]
    ]
    if failures:
        lines += ["", "Named failures (the first per robot and check):", ""]
        for robot, row in failures:
            detail = row["detail"].replace("|", "/").replace("\n", " ")
            lines.append(f"- `{robot}` / {row['check']}: {detail[:220]}")
    return lines


def render_parity(pair: str, date: str, files: list[Path]) -> list[str]:
    """The ``robot parity`` runs of one engine pair, one row per result file."""
    lines = [
        f"### Cross-engine parity: {pair.replace('-', ' vs ')} ({date})",
        "",
        "`python -m metasim robot parity <robot> --sims "
        + pair.replace("-", " ")
        + "`: the same canned joint trajectory on both "
        f"engines, judged per joint. Source: `results/robots/{date}-parity-{pair}/`.",
        "",
        "| robot | run | control steps | peak speed | worst joint | max diff | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in sorted(files):
        d = json.loads(path.read_text())
        rows = d.get("rows", [])
        worst = max(rows, key=lambda r: r.get("max_diff", 0.0)) if rows else None
        unit = "m" if worst and worst.get("kind") == "slide" else "rad"
        if not d.get("judged", True):
            verdict = "reported, not judged (floating base)"
        else:
            verdict = "pass" if d.get("ok") else "FAIL"
        peak = f"{d['peak_velocity']:.2f}/s" if d.get("peak_velocity") is not None else "-"
        run = path.stem.split("__", 1)[1].replace("_", " ") if "__" in path.stem else "default"
        lines.append(
            f"| `{d.get('robot', path.stem)}` | {run} | {d.get('steps', '-')} | {peak} | "
            + (f"`{worst['joint']}` | {worst['max_diff']:.4f} {unit} | " if worst else "- | - | ")
            + f"{verdict} |"
        )
    return lines


def render(results: Path = RESULTS) -> str:
    """The whole page."""
    rows = config_rows()
    lines = [
        "# Robots",
        "",
        "Generated from `roboverse_pack/robots` and `results/robots/` by `generate_robot_docs.py`; do not edit by hand.",
        "A robot is listed with the asset formats its config declares; `tools/assets/check_manifest.py` verifies on",
        "every pull request that each declared file exists on the Hub. Robots whose assets did not exist were retired",
        "on 2026-11-28 (`roboverse_pack/tasks/RETIRED_TASKS.md`). The self-check and parity tables come from the newest",
        "result files per simulator; how to add a robot and pass them is in [Add a robot](../../start/add_a_robot.md).",
        "",
        "## Configs",
        "",
        "| config module | class | `name` | joints | asset formats |",
        "|---|---|---|---|---|",
    ]
    for mod_name, cls_name, joints, assets, name in rows:
        lines.append(f"| `{mod_name}` | `{cls_name}` | `{name}` | {joints} | {assets} |")
    lines += ["", f"{sum(1 for r in rows if r[1])} robot configs."]
    checks = latest_dirs(_CHECK_DIR, results)
    if checks:
        lines += ["", "## Self-check", ""]
        for sim, (date, d) in sorted(checks.items()):
            path = d / "robots.json"
            if path.exists():
                lines += render_check(sim, date, json.loads(path.read_text())) + [""]
    parity = latest_dirs(_PARITY_DIR, results)
    if parity:
        lines += ["## Cross-engine parity", ""]
        for pair, (date, d) in sorted(parity.items()):
            lines += render_parity(pair, date, list(d.glob("*.json"))) + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail when the committed page differs from the generated one")
    args = ap.parse_args(argv)
    out = Path(__file__).with_name("robots.md")
    text = render()
    if args.check:
        current = out.read_text() if out.exists() else ""
        if current != text:
            sys.stderr.write(f"{out} is stale: run python {Path(__file__).relative_to(ROOT)}\n")
            return 1
        sys.stderr.write(f"{out}: up to date\n")
        return 0
    out.write_text(text)
    sys.stderr.write(f"{out}: {text.count(chr(10))} lines\n")
    return 0


if __name__ == "__main__":
    from metasim.sim.gl import apply_headless_gl

    apply_headless_gl()
    sys.exit(main())
