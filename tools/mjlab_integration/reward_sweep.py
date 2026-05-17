"""Compute mjlab reward curves on raw + metasim rollouts and plot.

For each task in the inventory that has a reward port available, load the
two saved rollouts, evaluate the reward, and emit a comparison plot at
`runs/<task>/reward_compare.png` plus a summary JSON.

Run after `tools.mjlab_integration.full_runner` / `render_sweep` so the
rollouts are persisted as `runs/<task>/raw_mujoco.npz` and
`runs/<task>/metasim_full.npz`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import inventory as inv
from . import rewards as rew
from .render_sweep import _load_npz

REWARD_FNS = {
    "mjlab.cartpole_balance": rew.cartpole_reward,
    "mjlab.cartpole_swingup": rew.cartpole_reward,
}


def _plot(raw_r: np.ndarray, meta_r: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(raw_r.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    axes[0].plot(t, raw_r, lw=1.6, label="raw mujoco")
    axes[0].plot(t, meta_r, lw=1.0, ls="--", label="MetaSim full")
    axes[0].set_ylabel("reward")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    diff = np.abs(raw_r - meta_r)
    axes[1].plot(t, diff, lw=1.2, color="firebrick")
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("|raw - meta|")
    axes[1].set_yscale("symlog", linthresh=1e-10)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/ghr/projects/RoboVerse/reports/mjlab_integration")
    args = p.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    out_dir = Path(args.out)
    summary: dict = {"rewards": []}

    for task in inv.ALL_TASKS:
        fn = REWARD_FNS.get(task.metasim_name)
        if fn is None:
            continue
        run_dir = out_dir / "runs" / task.metasim_name
        raw = _load_npz(run_dir / "raw_mujoco.npz")
        meta = _load_npz(run_dir / "metasim_full.npz")
        if raw is None or meta is None:
            print(f"[reward_sweep] {task.metasim_name}: missing npz; skip")
            continue

        try:
            raw_r = fn(raw)
            meta_r = fn(meta)
        except Exception as e:
            print(f"[reward_sweep] {task.metasim_name}: reward fn error: {e}")
            summary["rewards"].append({"task": task.metasim_name, "error": str(e)})
            continue

        diff = np.abs(raw_r - meta_r)
        entry = {
            "task": task.metasim_name,
            "reward_n": int(raw_r.shape[0]),
            "raw_sum": float(raw_r.sum()),
            "meta_sum": float(meta_r.sum()),
            "raw_mean": float(raw_r.mean()),
            "meta_mean": float(meta_r.mean()),
            "max_abs_diff": float(diff.max()),
            "rms_diff": float(np.sqrt((diff**2).mean())),
        }
        out_png = run_dir / "reward_compare.png"
        _plot(raw_r, meta_r, out_png, title=f"{task.metasim_name} — mjlab reward")
        entry["plot"] = str(out_png)
        summary["rewards"].append(entry)
        print(
            f"[reward_sweep] {task.metasim_name}  sum raw={entry['raw_sum']:.3f} "
            f"meta={entry['meta_sum']:.3f}  max|Δ|={entry['max_abs_diff']:.3e}"
        )

    summary_path = out_dir / "reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[reward_sweep] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
