"""Engine-parity sweep: robosuite-native vs MetaSim's dm_control loader.

For each task and control signal we step the *same* compiled robosuite MJCF on
both engines and quantify divergence. Writes per-task npz + a machine-readable
``engine_parity.json`` the report renders into a table.

Usage:
    python -m tools.robosuite_integration.runner \
        --out /workspace/Project/research_report/projects/roboverse/robosuite_1to1/runs \
        --n-steps 100 --tasks Lift Stack Door PickPlaceCan NutAssemblySquare
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

# robosuite spams part-controller warnings for single-arm robots; quiet them.
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

from .common import sinusoid_ctrl, zero_ctrl
from .diff import diff_rollouts
from .inventory import ALL_TASKS, get
from .metasim_rollout import metasim_dm_control_rollout
from .robosuite_rollout import robosuite_engine_rollout

CTRL_BUILDERS = {
    "zero": lambda nu: zero_ctrl(nu),
    "sinusoid": lambda nu: sinusoid_ctrl(nu, amp=0.5, period=40),
}


def run_task(task_name: str, *, n_steps: int, out_dir: Path, signals: list[str]) -> dict:
    task = get(task_name)
    run_dir = out_dir / task.slug
    run_dir.mkdir(parents=True, exist_ok=True)
    entry: dict = {"task": task.metasim_name, "env_name": task.env_name, "robot": task.robot, "signals": {}}

    for sig in signals:
        # The same nu-aware builder feeds both engines, so the ctrl signals match
        # exactly regardless of the model's actuator count.
        t0 = time.time()
        builder = CTRL_BUILDERS[sig]
        rs, xml, overrides = robosuite_engine_rollout(task, n_steps=n_steps, ctrl_builder=builder, seed=0)
        nu = rs.ctrl.shape[1]
        ms = metasim_dm_control_rollout(
            xml,
            task_id=rs.task_id,
            n_steps=n_steps,
            decimation=rs.decimation,
            ctrl_fn=builder(nu),
            qpos0=rs.qpos[0],
            qvel0=rs.qvel[0],
            model_overrides=overrides,
        )
        ds = diff_rollouts(rs, ms)
        rs.save(run_dir / f"robosuite_native__{sig}.npz")
        ms.save(run_dir / f"metasim_dm_control__{sig}.npz")
        dt = time.time() - t0
        s = ds.to_dict()
        s["wall_s"] = round(dt, 2)
        s["nu"] = nu
        # how much runtime model-mutation get_xml() omitted (0 for free-joint placement)
        import numpy as np

        s["model_override_body_pos_max"] = float(np.abs(overrides["body_pos"]).max())
        entry["signals"][sig] = s
        print(
            f"  [{task.env_name:18s} {sig:9s}] nq={ds.nq:3d} nv={ds.nv:3d} nu={nu:2d} "
            f"qpos_max={ds.qpos_max_abs:.2e} qvel_max={ds.qvel_max_abs:.2e} "
            f"bitwise={ds.bitwise_equal} ({dt:.1f}s)"
        )
    (run_dir / "engine_parity.json").write_text(json.dumps(entry, indent=2))
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--tasks", nargs="*", default=[t.env_name for t in ALL_TASKS])
    ap.add_argument("--signals", nargs="*", default=["zero", "sinusoid"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"n_steps": args.n_steps, "tasks": []}
    print(f"engine-parity sweep: {len(args.tasks)} tasks x {len(args.signals)} signals, {args.n_steps} steps")
    for name in args.tasks:
        try:
            entry = run_task(name, n_steps=args.n_steps, out_dir=args.out, signals=args.signals)
            summary["tasks"].append(entry)
        except Exception as e:  # keep the sweep going; record the failure
            import traceback

            traceback.print_exc()
            summary["tasks"].append({"task": name, "error": str(e)})

    (args.out.parent / "engine_parity.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out.parent / 'engine_parity.json'}")


if __name__ == "__main__":
    main()
