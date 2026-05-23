"""Plot MetaSim-native training curves (cartpole) with mjlab reference overlay.

Supports plotting multiple runs (v3, v4, etc.) for ablation comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")

_TXT_PATTERN = __import__("re").compile(
    r"\[iter\s+(\d+)/\d+\]\s+steps=\s*\d+\s+step_r=([\d.\-]+)\s+ep_r=\s*([\d.\-nan]+)"
    r"\s+ep_len=\s*([\d.\-nan]+)\s+pg=([\d.\-]+)\s+v=([\d.\-]+)\s+ent=([\d.\-]+)"
)


def load_log(path: Path) -> dict[str, np.ndarray]:
    """Accept either JSONL (preferred) or the stdout text format (fallback)."""
    text = path.read_text()
    rows = []
    if path.suffix == ".jsonl" or (text and text.lstrip().startswith("{")):
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        # Fall back to parsing the stdout `[iter N/M]` lines.
        for line in text.splitlines():
            m = _TXT_PATTERN.search(line)
            if not m:
                continue
            rows.append({
                "iter": int(m.group(1)),
                "mean_step_reward": float(m.group(2)),
                "mean_reward_recent": float("nan") if "nan" in m.group(3) else float(m.group(3)),
                "mean_ep_len_recent": float("nan") if "nan" in m.group(4) else float(m.group(4)),
                "pg_loss": float(m.group(5)),
                "v_loss": float(m.group(6)),
                "entropy": float(m.group(7)),
            })
    return {
        "iters": np.array([r["iter"] for r in rows]),
        "ep_r": np.array([r.get("mean_reward_recent", float("nan")) for r in rows]),
        "step_r": np.array([r.get("mean_step_reward", float("nan")) for r in rows]),
        "pg": np.array([r.get("pg_loss", float("nan")) for r in rows]),
        "v": np.array([r.get("v_loss", float("nan")) for r in rows]),
        "ent": np.array([r.get("entropy", float("nan")) for r in rows]),
        "lr": np.array([r.get("lr", float("nan")) for r in rows]),
        "kl": np.array([r.get("kl", float("nan")) for r in rows]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--logs",
        nargs="+",
        default=[
            "/tmp/metasim_train_cartpole_v3.log_jsonl",  # backed up earlier; may not exist
            os.path.join("training_logs", "metasim_native/mjlab_cartpole_balance_train/training_log.jsonl"),
        ],
    )
    p.add_argument("--labels", nargs="+", default=None, help="Per-log display labels. If None, derives from path.")
    p.add_argument(
        "--out",
        default=os.path.join(
            _REPORTS, "mjlab_integration/runs/mjlab.cartpole_balance/metasim_native_training_curve.png"
        ),
    )
    p.add_argument("--mjlab-reference", type=float, default=49.93)
    p.add_argument("--title", default="MetaSim-native cartpole_balance training (in-framework PPO)")
    args = p.parse_args(argv)

    # Load all available logs
    logs = []
    for i, lp in enumerate(args.logs):
        p_ = Path(lp)
        if not p_.exists():
            print(f"  [skip] {lp} (not found)")
            continue
        label = args.labels[i] if args.labels and i < len(args.labels) else p_.parent.name
        logs.append((label, load_log(p_)))
    if not logs:
        print("no logs to plot")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle(args.title, fontsize=13)

    colors = ["steelblue", "darkorange", "green", "purple"]

    ax = axes[0, 0]
    for (label, log), c in zip(logs, colors):
        ax.plot(log["iters"], log["ep_r"], lw=1.5, color=c, label=label)
    ax.axhline(
        args.mjlab_reference, color="red", linestyle="--", lw=1, label=f"mjlab native ({args.mjlab_reference:.2f})"
    )
    ax.axhline(50.0, color="grey", linestyle=":", lw=1, label="theoretical max (50)")
    ax.set_title("Mean episode reward (recent 64 episodes)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("ep_return")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for (label, log), c in zip(logs, colors):
        # smooth step_r with rolling mean for readability
        x = log["iters"]
        y = log["step_r"]
        if len(y) > 20:
            w = max(5, len(y) // 50)
            y = np.convolve(y, np.ones(w) / w, mode="same")
        ax.plot(x, y, lw=1.0, color=c, alpha=0.8, label=label)
    ax.set_title("Per-step reward (smoothed)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("step_r")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for (label, log), c in zip(logs, colors):
        ax.plot(log["iters"], log["v"], lw=1.0, color=c, label=label)
    ax.set_title("Value loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("v_loss")
    ax.set_yscale("symlog")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for (label, log), c in zip(logs, colors):
        ax.plot(log["iters"], log["ent"], lw=1.0, color=c, label=label)
    ax.set_title("Entropy (action distribution)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("entropy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    for label, log in logs:
        if len(log["ep_r"]) > 0 and not np.isnan(log["ep_r"][-1]):
            print(
                f"  [{label}] iter={log['iters'][-1]:>5} ep_r={log['ep_r'][-1]:.2f} "
                f"({log['ep_r'][-1] / args.mjlab_reference * 100:.1f}% of mjlab)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
