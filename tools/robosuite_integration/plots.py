"""Generate the report's parity charts from the saved npz / json artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import RolloutResult
from .inventory import ALL_TASKS

_BLUE = "#3b82f6"
_ORANGE = "#f59e0b"
_BG = "#0e1014"
_FG = "#e6e8ee"


def _style(ax):
    ax.set_facecolor(_BG)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    ax.grid(True, color="#23262d", lw=0.6)


def qpos_overlay(run_dir: Path, signal: str = "sinusoid") -> Path | None:
    a = run_dir / f"robosuite_native__{signal}.npz"
    b = run_dir / f"metasim_dm_control__{signal}.npz"
    if not (a.exists() and b.exists()):
        return None
    ra, rb = RolloutResult.load(a), RolloutResult.load(b)
    n_show = min(6, ra.qpos.shape[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), facecolor=_BG, height_ratios=[2, 1])
    t = np.arange(ra.qpos.shape[0])
    for j in range(n_show):
        ax1.plot(t, ra.qpos[:, j], color=_BLUE, lw=2.4, alpha=0.85)
        ax1.plot(t, rb.qpos[:, j], color=_ORANGE, lw=0.9, ls="--")
    ax1.plot([], [], color=_BLUE, lw=2.4, label="robosuite native MjSim")
    ax1.plot([], [], color=_ORANGE, lw=0.9, ls="--", label="MetaSim MujocoHandler")
    ax1.set_title(f"{ra.task_id} — qpos trajectories ({signal} ctrl), first {n_show} DOF")
    ax1.set_ylabel("qpos")
    ax1.legend(facecolor="#16181d", edgecolor="#444", labelcolor=_FG, fontsize=8, loc="upper right")
    _style(ax1)
    diff = np.abs(ra.qpos - rb.qpos).max(axis=1)
    ax2.plot(t, diff, color="#10b981", lw=1.5)
    ax2.set_title(f"max|Δqpos| per step  (peak = {diff.max():.1e})")
    ax2.set_xlabel("control step")
    ax2.set_ylabel("abs diff")
    _style(ax2)
    fig.tight_layout()
    out = run_dir / f"qpos_overlay_{signal}.png"
    fig.savefig(out, dpi=120, facecolor=_BG)
    plt.close(fig)
    return out


def reward_curve(run_dir: Path) -> Path | None:
    pj = run_dir / "replay_parity.json"
    if not pj.exists():
        return None
    # we only stored aggregates; recompute cumulative from rollouts is not saved,
    # so draw the matching return as a simple cumulative line from the json totals.
    data = json.loads(pj.read_text())
    return None if "error" in data else _draw_return_bar(run_dir, data)


def _draw_return_bar(run_dir: Path, data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor=_BG)
    vals = [data["ref_total_reward"], data["metasim_total_reward"]]
    ax.bar(["robosuite", "MetaSim"], vals, color=[_BLUE, _ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", color=_FG, fontsize=9)
    ax.set_title(f"{data['task']} — episode return")
    _style(ax)
    fig.tight_layout()
    out = run_dir / "return_bar.png"
    fig.savefig(out, dpi=120, facecolor=_BG)
    plt.close(fig)
    return out


def parity_overview(report_dir: Path) -> Path | None:
    """Parity matrix: rows = tasks, columns = checks, each cell shows max|Δqpos|."""
    ep = report_dir / "engine_parity.json"
    rp = report_dir / "replay_parity.json"
    if not ep.exists():
        return None
    e = json.loads(ep.read_text())
    rep = {t["task"]: t for t in json.loads(rp.read_text())["tasks"]} if rp.exists() else {}
    cols = ["engine\n(zero ctrl)", "engine\n(sinusoid)", "trajectory\nreplay", "reward\n& success"]
    rows, cells = [], []
    for entry in e["tasks"]:
        if "error" in entry:
            continue
        rows.append(entry["env_name"])
        r = rep.get(entry["task"], {})
        ok_rs = r.get("reward_bitwise_equal", False) and r.get("success_sequence_match", False)
        cells.append([
            entry["signals"]["zero"]["qpos_max_abs"],
            entry["signals"]["sinusoid"]["qpos_max_abs"],
            r.get("qpos_max_abs_diff", float("nan")),
            0.0 if ok_rs else float("nan"),
        ])
    cells = np.array(cells)
    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(rows) + 1.6), facecolor=_BG)
    ax.set_facecolor(_BG)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = cells[i, j]
            ok = (v == 0.0) or (v <= 1e-12)
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color="#10532f" if ok else "#5b1d1d"))
            txt = "0.0 ✓" if v == 0.0 else (f"{v:.0e}" if not np.isnan(v) else "—")
            ax.text(j + 0.48, i + 0.46, txt, ha="center", va="center", color=_FG, fontsize=9, weight="bold")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.48)
    ax.set_xticklabels(cols, fontsize=8.5, color=_FG)
    ax.set_yticks(np.arange(len(rows)) + 0.46)
    ax.set_yticklabels(rows, fontsize=9, color=_FG)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("robosuite ↔ MetaSim MuJoCo · max |Δqpos| (bitwise = 0.0)", color=_FG, pad=26)
    fig.tight_layout()
    out = report_dir / "parity_overview.png"
    fig.savefig(out, dpi=120, facecolor=_BG)
    plt.close(fig)
    return out


def policy_eval_scatter(report_dir: Path) -> Path | None:
    pj = report_dir / "policy_eval.json"
    if not pj.exists():
        return None
    aggs = json.loads(pj.read_text())["tasks"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=_BG)
    for agg in aggs:
        eps = agg["episodes"]
        rr = [e["ref_return"] for e in eps]
        mr = [e["metasim_return"] for e in eps]
        ax.scatter(rr, mr, s=60, color=_ORANGE, edgecolor=_FG, zorder=3, label=agg["task"])
    lo = min([e["ref_return"] for agg in aggs for e in agg["episodes"]] + [0])
    hi = max([e["ref_return"] for agg in aggs for e in agg["episodes"]] + [1])
    ax.plot([lo, hi], [lo, hi], color=_BLUE, lw=1.2, ls="--", label="y = x (identical)")
    ax.set_xlabel("robosuite episode return")
    ax.set_ylabel("MetaSim replay return")
    ax.set_title("Per-seed return: robosuite vs MetaSim")
    ax.legend(facecolor="#16181d", edgecolor="#444", labelcolor=_FG, fontsize=8)
    _style(ax)
    fig.tight_layout()
    out = report_dir / "policy_eval_scatter.png"
    fig.savefig(out, dpi=120, facecolor=_BG)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    runs = args.report / "runs"
    made = []
    for task in ALL_TASKS:
        rd = runs / task.slug
        if rd.exists():
            for sig in ("sinusoid", "zero"):
                p = qpos_overlay(rd, sig)
                if p:
                    made.append(p)
            p = reward_curve(rd)
            if p:
                made.append(p)
    for fn in (parity_overview, policy_eval_scatter):
        p = fn(args.report)
        if p:
            made.append(p)
    print(f"generated {len(made)} charts")
    for p in made:
        print("  ", p.relative_to(args.report))


if __name__ == "__main__":
    main()
