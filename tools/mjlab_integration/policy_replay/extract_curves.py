"""Read RSL-RL tensorboard event files; emit per-task reward/loss curves + summary.

Used to demonstrate that mjlab-task training converges in our setup (post the
1:1 task port). Writes:
  reports/mjlab_integration/runs/<task>/training_curve.png
  reports/mjlab_integration/training_summary.json

If multiple run dirs exist for a task (different seeds / dates), pick the
latest by mtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# experiment_name → metasim task name (for output dir)
TASK_MAP = {
    "cartpole_balance": "mjlab.cartpole_balance",
    "cartpole_swingup": "mjlab.cartpole_swingup",
    "g1_velocity": "mjlab.velocity_flat_g1",
    "g1_velocity_rough": "mjlab.velocity_rough_g1",
    "go1_velocity_flat": "mjlab.velocity_flat_go1",
    "go1_velocity_rough": "mjlab.velocity_rough_go1",
    "g1_tracking_no_state_est": "mjlab.tracking_flat_g1_no_state_est",
    "yam_lift_cube": "mjlab.lift_cube_yam",
    "yam_lift_cube_rgb": "mjlab.lift_cube_yam_rgb",
    "yam_lift_cube_depth": "mjlab.lift_cube_yam_depth",
    "yam_multi_cube": "mjlab.multi_cube_seg_yam",
}


def latest_run_dir(exp_dir: Path) -> Path | None:
    candidates = [d for d in exp_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for all scalar tags."""
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0 = no limit
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in tags:
        out[t] = [(e.step, e.value) for e in ea.Scalars(t)]
    return out


def plot_run(run_dir: Path, task_name: str, out_png: Path, scalars: dict) -> dict:
    """Make a 2x2 panel: mean reward, value loss, mean episode length, action std."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"{task_name}  — training curve (RSL-RL PPO)", fontsize=13)

    def _plot(ax, candidates, title, ylabel, scale_y=False):
        for cand in candidates:
            if cand in scalars:
                xs, ys = zip(*scalars[cand])
                ax.plot(xs, ys, lw=1.5, label=cand.split("/")[-1])
                break
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("iter")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if scale_y:
            ax.set_yscale("symlog")

    _plot(
        axes[0, 0], ["Train/mean_reward", "rollout/ep_rew_mean", "Train/Mean Reward"], "Mean episode reward", "reward"
    )
    _plot(
        axes[0, 1], ["Loss/value_function", "Loss/value", "train/value_loss"], "Value loss", "value loss", scale_y=True
    )
    _plot(axes[1, 0], ["Train/mean_episode_length", "rollout/ep_len_mean", "Episode/Length"], "Episode length", "steps")
    _plot(
        axes[1, 1], ["Train/mean_noise_std", "Loss/learning_rate", "Policy/lr"], "Action noise / learning rate", "value"
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Summary stats: final reward + iter count
    summary = {"task": task_name, "run_dir": str(run_dir)}
    for r in ("Train/mean_reward", "rollout/ep_rew_mean", "Train/Mean Reward"):
        if scalars.get(r):
            xs, ys = zip(*scalars[r])
            summary["iter_final"] = int(xs[-1])
            summary["reward_final"] = float(ys[-1])
            summary["reward_max"] = float(np.max(ys))
            summary["reward_mean_last10pct"] = float(np.mean(ys[-max(1, len(ys) // 10) :]))
            break
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logs-root", default="/home/ghr/projects/RoboVerse/RoboVerse/training_logs/rsl_rl")
    p.add_argument("--runs-root", default="/home/ghr/projects/RoboVerse/reports/mjlab_integration/runs")
    p.add_argument(
        "--summary-out", default="/home/ghr/projects/RoboVerse/reports/mjlab_integration/training_summary.json"
    )
    args = p.parse_args(argv)

    logs_root = Path(args.logs_root)
    runs_root = Path(args.runs_root)
    all_summaries = []

    for exp_name, task_name in TASK_MAP.items():
        exp_dir = logs_root / exp_name
        if not exp_dir.exists():
            print(f"  [skip] {exp_name}: no log dir")
            continue
        run_dir = latest_run_dir(exp_dir)
        if run_dir is None:
            print(f"  [skip] {exp_name}: no run sub-dir under {exp_dir}")
            continue

        try:
            scalars = load_scalars(run_dir)
        except Exception as e:
            print(f"  [error] {exp_name}: {e}")
            continue

        if not scalars:
            print(f"  [skip] {exp_name}: no scalars in {run_dir}")
            continue

        out_png = runs_root / task_name / "training_curve.png"
        summary = plot_run(run_dir, task_name, out_png, scalars)
        all_summaries.append(summary)
        print(
            f"  [{exp_name}] iter={summary.get('iter_final', '?')} "
            f"final={summary.get('reward_final', float('nan')):.3f} "
            f"max={summary.get('reward_max', float('nan')):.3f} "
            f"→ {out_png}"
        )

    Path(args.summary_out).write_text(json.dumps(all_summaries, indent=2))
    print(f"\nwrote summary {args.summary_out}  ({len(all_summaries)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
