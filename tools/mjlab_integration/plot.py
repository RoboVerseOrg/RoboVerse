"""Generate PNG comparison plots for a pair of rollouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .raw_rollout import RolloutResult


def plot_trajectory_pair(a: RolloutResult, b: RolloutResult, out_path: Path, title: str | None = None) -> None:
    """Plot qpos[:n_show] overlaid for a (solid) vs b (dashed) + abs diff below."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nq = a.qpos.shape[1]
    n_show = min(nq, 6)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))

    for i in range(n_show):
        label_a = a.joint_names[i] if i < len(a.joint_names) else f"q{i}"
        axes[0].plot(a.time, a.qpos[:, i], color=colors[i], lw=1.5, label=f"{label_a} ({a.backend})")
        axes[0].plot(b.time, b.qpos[:, i], color=colors[i], lw=1.0, ls="--", label=f"{label_a} ({b.backend})")

    axes[0].set_ylabel("qpos")
    axes[0].legend(fontsize=7, loc="best", ncol=2)
    if title is not None:
        axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    diff = np.abs(a.qpos - b.qpos)
    for i in range(n_show):
        label_a = a.joint_names[i] if i < len(a.joint_names) else f"q{i}"
        axes[1].plot(a.time, diff[:, i], color=colors[i], lw=1.0, label=label_a)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel(f"|qpos_{{ {a.backend} }} - qpos_{{ {b.backend} }}|")
    axes[1].set_yscale("symlog", linthresh=1e-10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_ctrl(result: RolloutResult, out_path: Path, title: str | None = None) -> None:
    """Plot the ctrl applied per control step."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if result.ctrl.size == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 3.5))
    nu = result.ctrl.shape[1]
    for i in range(nu):
        name = result.actuator_names[i] if i < len(result.actuator_names) else f"u{i}"
        ax.plot(result.ctrl[:, i], lw=1.0, label=name)
    ax.set_xlabel("control step")
    ax.set_ylabel("ctrl")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=min(nu, 6), loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
