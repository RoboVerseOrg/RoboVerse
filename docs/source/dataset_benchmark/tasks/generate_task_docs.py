"""Regenerate inventory.md from the task registry (no hand-written rows).

    python docs/source/dataset_benchmark/tasks/generate_task_docs.py           # rewrite the page
    python docs/source/dataset_benchmark/tasks/generate_task_docs.py --check   # exit 1 if the page is stale (CI)

The page lists every registered task name with the module that registers it, grouped by pack, from MetaSim's static
registry index (nothing is imported), plus the suites that score them and the retired list. A task appears here
because it is registered, nowhere else; the gate decides whether it stays.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "metasim"))

RETIRED_CSV = ROOT / "roboverse_pack" / "tasks" / "retired_tasks.csv"


def pack_of(module: str) -> str:
    """``roboverse_pack.tasks.maniskill.pick_cube`` -> ``maniskill``; a module outside the pack keeps its root."""
    parts = module.split(".")
    if parts[:2] == ["roboverse_pack", "tasks"] and len(parts) > 2:
        return parts[2]  # a module directly under tasks/ (task_template) is its own "pack"
    if parts[:1] == ["metasim"]:
        return "metasim example pack"
    return parts[0]


def registered() -> dict[str, str]:
    """``{task name: registering module}`` for every registered task, from the static index."""
    from metasim.task.registry import _static_index, list_tasks

    index = _static_index()
    out = {}
    for name in list_tasks():
        out[name] = index.names.get(name.lower(), "(registered at import time)")
    return out


def suites() -> dict[str, dict[str, list[str]]]:
    """``{suite: {split: [tasks]}}`` for the registered suites."""
    from metasim.bench import get_suite, list_suites

    return {name: {s: [t.task for t in ts] for s, ts in get_suite(name).splits.items()} for name in list_suites()}


def retired_rows() -> list[dict[str, str]]:
    if not RETIRED_CSV.exists():
        return []
    with RETIRED_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def render() -> str:
    """The whole page."""
    tasks = registered()
    by_pack: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, module in tasks.items():
        by_pack[pack_of(module)].append((name, module))
    lines = [
        "# Task inventory",
        "",
        "Generated from the task registry by `generate_task_docs.py`; do not edit by hand. Every name below is",
        "registered in this checkout (`python -m metasim bench list` prints the same list); the release gate",
        "(`bench health --floor --repeat 2`, nightly on MuJoCo and Isaac Sim 6) decides whether a task stays, and",
        "`roboverse_pack/tasks/RETIRED_TASKS.md` records the ones it removed and the commit to restore them from.",
        "How to add one: [Add a task](../../start/add_a_task.md).",
        "",
        f"{len(tasks)} registered task names in {len(by_pack)} packs.",
        "",
        "## Suites",
        "",
    ]
    for suite, splits in sorted(suites().items()):
        for split, names in splits.items():
            lines.append(f"- `{suite}` / `{split}`: " + ", ".join(f"`{n}`" for n in names))
    lines += ["", "## Registered tasks by pack", ""]
    for pack in sorted(by_pack):
        rows = sorted(by_pack[pack])
        lines += [f"### {pack} ({len(rows)})", "", "| task | registered in |", "|---|---|"]
        lines += [f"| `{name}` | `{module}` |" for name, module in rows]
        lines.append("")
    retired = retired_rows()
    lines += ["## Retired", ""]
    if retired:
        seen: dict[str, str] = {}
        for row in retired:
            seen.setdefault(row.get("task", ""), row.get("family", "?"))
        by_family = defaultdict(int)
        for family in seen.values():
            by_family[family] += 1
        lines.append(
            f"{len(seen)} retired tasks (`roboverse_pack/tasks/retired_tasks.csv`, with the commit to restore each from; "
            "`RETIRED_TASKS.md` tells the story), by family: "
            + ", ".join(f"{f} ({n})" for f, n in sorted(by_family.items()))
            + "."
        )
    else:
        lines.append("No retired-task list in this checkout.")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail when the committed page differs from the generated one")
    args = ap.parse_args(argv)
    out = Path(__file__).with_name("inventory.md")
    text = render()
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != text:
            sys.stderr.write(f"{out} is stale: run python {Path(__file__).relative_to(ROOT)}\n")
            return 1
        sys.stderr.write(f"{out}: up to date\n")
        return 0
    out.write_text(text, encoding="utf-8")
    sys.stderr.write(f"{out}: {text.count(chr(10))} lines\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
