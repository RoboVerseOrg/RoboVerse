"""Sweep every RoboTwin task through the data bridge and record a coverage matrix.

Runs in the dedicated ``robotwin`` conda env. For each task discovered under
``$ROBOTWIN_DIR/envs``, it shells out to ``collect_bridge.py`` (one isolated
process per task, so a SAPIEN renderer crash in one task cannot poison the
others) and records whether the native RoboTwin task planned + checked
successfully and how many frames the resulting bridge trajectory holds.

The result is a JSON coverage report written incrementally (so a long sweep is
resumable / inspectable mid-run) plus an end-of-run summary. This is the breadth
evidence for the RoboTwin integration: it shows the passthrough + bridge work
across the *whole* task suite, not just one hand-picked task.

Run::

    conda run -n robotwin env MUJOCO_GL=egl SAPIEN_HEADLESS=1 \\
        python tools/robotwin_integration/coverage_sweep.py \\
        --max-seeds 8 --out outputs/robotwin_coverage/coverage.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_BRIDGE = os.path.join(_HERE, "collect_bridge.py")
_PASSTHROUGH = os.path.join(_REPO_ROOT, "roboverse_pack", "tasks", "robotwin", "_passthrough.py")

_SAVED_RE = re.compile(r"SAVED (\d+) frames \(seed (\d+)\)")


def _load_passthrough():
    spec = importlib.util.spec_from_file_location("rt_passthrough", _PASSTHROUGH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_one(task: str, work_dir: str, max_seeds: int, save_freq: int, timeout: int) -> dict:
    out_pkl = os.path.join(work_dir, f"{task}.pkl")
    cmd = [
        sys.executable, _BRIDGE,
        "--task", task,
        "--max-seeds", str(max_seeds),
        "--save-freq", str(save_freq),
        "--out", out_pkl,
    ]  # fmt: skip
    env = dict(os.environ, MUJOCO_GL="egl", SAPIEN_HEADLESS="1")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {"task": task, "status": "timeout", "seconds": round(time.time() - t0, 1)}
    dt = round(time.time() - t0, 1)
    out = proc.stdout + "\n" + proc.stderr
    m = _SAVED_RE.search(out)
    if m:
        return {
            "task": task, "status": "ok", "frames": int(m.group(1)),
            "seed": int(m.group(2)), "pkl": out_pkl, "seconds": dt,
        }  # fmt: skip
    if "NO_SUCCESS" in out:
        return {"task": task, "status": "no_success", "seconds": dt}
    # last non-empty, non-noise line is the most informative error
    tail = [
        ln for ln in out.splitlines()
        if ln.strip() and "svulkan2" not in ln and "OIDN" not in ln and "pkg_resources" not in ln
    ]  # fmt: skip
    return {"task": task, "status": "error", "seconds": dt, "tail": tail[-3:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-seeds", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=600, help="per-task wall-clock cap (s)")
    ap.add_argument("--only", nargs="*", help="restrict to these task names")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "outputs", "robotwin_coverage", "coverage.json"))
    args = ap.parse_args(argv)

    pt = _load_passthrough()
    rt = pt.robotwin_dir()
    tasks = args.only or pt._discover_task_names(rt)
    if not tasks:
        print(f"No RoboTwin tasks found under {rt}/envs")
        return 1
    work_dir = os.path.join(rt, "data", "_rv_bridge")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Sweeping {len(tasks)} RoboTwin tasks (max_seeds={args.max_seeds}) -> {args.out}")
    results: list[dict] = []
    for i, task in enumerate(tasks):
        print(f"\n[{i + 1}/{len(tasks)}] {task} ...", flush=True)
        r = _run_one(task, work_dir, args.max_seeds, args.save_freq, args.timeout)
        results.append(r)
        print(f"  -> {r['status']} ({r.get('frames', '-')} frames, {r['seconds']}s)", flush=True)
        n_ok = sum(1 for x in results if x["status"] == "ok")
        summary = {
            "total": len(tasks), "done": len(results), "ok": n_ok,
            "max_seeds": args.max_seeds, "results": results,
        }  # fmt: skip
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)

    n_ok = sum(1 for x in results if x["status"] == "ok")
    print(f"\n=== COVERAGE: {n_ok}/{len(tasks)} tasks produced a successful bridge trajectory ===")
    for x in results:
        if x["status"] != "ok":
            print(f"  {x['status']:>10}  {x['task']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
