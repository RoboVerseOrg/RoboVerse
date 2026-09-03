"""Cross-sim joint-target tracking parity: SuperDex vs MuJoCo (or any two backends).

Runs the ``get_started/1_control_robot.py`` scene with a *seeded* random joint-target sequence on one
backend and records, per env step, the commanded targets and the measured joint positions. Because
SuperDex ships Python 3.12-only wheels while the MuJoCo env here is 3.11, each backend is recorded in
its own environment and the two recordings are compared afterwards::

    python scripts/parity_superdex_tracking.py record --sim mujoco   --out /tmp/parity/mujoco.npz
    python scripts/parity_superdex_tracking.py record --sim superdex --out /tmp/parity/superdex.npz
    python scripts/parity_superdex_tracking.py compare /tmp/parity/mujoco.npz /tmp/parity/superdex.npz --plot /tmp/parity/tracking.png

``compare`` prints, per backend, the mean / max absolute tracking error |q - q_target| over the run
and the mean absolute difference between the two backends' joint trajectories. It does not claim
parity: it measures it (see AGENTS.md, "Parity Is Load-Bearing").
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def _scenario(sim: str):
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
    from metasim.scenario.scenario import ScenarioCfg

    return ScenarioCfg(
        robots=["franka"],
        simulator=sim,
        headless=True,
        num_envs=1,
        objects=[
            PrimitiveCubeCfg(
                name="cube", size=(0.1, 0.1, 0.1), color=[1.0, 0.0, 0.0], physics=PhysicStateType.RIGIDBODY
            ),
            PrimitiveSphereCfg(name="sphere", radius=0.1, color=[0.0, 0.0, 1.0], physics=PhysicStateType.RIGIDBODY),
        ],
    )


def record(*, sim: str, out: str, steps: int, seed: int, hold: int) -> None:
    from metasim.utils.setup_util import get_handler

    scenario = _scenario(sim)
    handler = get_handler(scenario)
    robot = scenario.robots[0]
    joint_names = handler.get_joint_names(robot.name, sort=True)
    rng = np.random.default_rng(seed)
    init = {
        "objects": {
            "cube": {"pos": torch.tensor([0.3, -0.2, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
            "sphere": {"pos": torch.tensor([0.4, -0.6, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        },
        "robots": {
            robot.name: {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": dict(robot.default_joint_positions),
            }
        },
    }
    handler.set_states([init])

    targets, measured = [], []
    target = None
    for step in range(steps):
        if step % hold == 0:  # new random target every ``hold`` env steps, same sequence on every backend
            target = {
                jn: float(rng.uniform(robot.joint_limits[jn][0], robot.joint_limits[jn][1])) for jn in joint_names
            }
        handler.set_dof_targets([{robot.name: {"dof_pos_target": target}}])
        handler.simulate()
        state = handler.get_states(mode="dict")[0]["robots"][robot.name]["dof_pos"]
        targets.append([target[jn] for jn in joint_names])
        measured.append([float(state[jn]) for jn in joint_names])
    handler.close()

    out = out if out.endswith(".npz") else out + ".npz"  # np.savez appends the suffix itself
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, sim=sim, joint_names=np.array(joint_names), targets=np.array(targets), measured=np.array(measured))
    print(
        f"[{sim}] {steps} steps, hold={hold}: {_summary(joint_names, np.array(measured), np.array(targets))} -> {out}"
    )


def _summary(names: list[str], measured: np.ndarray, targets: np.ndarray) -> str:
    """Per-unit tracking error summary: revolute/arm joints (rad) and prismatic/finger joints (m) never mixed."""
    err = np.abs(measured - targets)
    arm = [j for j, n in enumerate(names) if "finger" not in n]
    fingers = [j for j, n in enumerate(names) if "finger" in n]
    parts = []
    if arm:
        parts.append(f"arm mean|err|={err[:, arm].mean():.4f} rad, max={err[:, arm].max():.4f} rad")
    if fingers:
        parts.append(
            f"finger mean|err|={err[:, fingers].mean() * 1000:.2f} mm, max={err[:, fingers].max() * 1000:.2f} mm"
        )
    return "; ".join(parts)


def compare(a_path: str, b_path: str, *, plot: str | None = None) -> None:
    a, b = np.load(a_path), np.load(b_path)
    names = list(a["joint_names"])
    if names != list(b["joint_names"]):
        raise ValueError(f"joint order differs: {names} vs {list(b['joint_names'])}")
    if a["targets"].shape != b["targets"].shape or not np.allclose(a["targets"], b["targets"]):
        raise ValueError("target sequences differ: record both backends with the same --seed/--hold/--steps")
    for rec in (a, b):
        print(f"[{rec['sim']}] {_summary(names, rec['measured'], rec['targets'])}")
    diff = np.abs(a["measured"] - b["measured"])
    print(f"[{a['sim']} vs {b['sim']}] {_summary(names, a['measured'], b['measured']).replace('err', 'Δq')}")
    for j, jn in enumerate(names):
        unit = "m" if "finger" in jn else "rad"
        print(f"  {jn:>20}: mean|Δq| = {diff[:, j].mean():.4f} {unit}  max = {diff[:, j].max():.4f} {unit}")
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = 3
        rows = -(-len(names) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3 * rows), sharex=True, squeeze=False)
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        t = np.arange(len(a["targets"]))
        for j, (ax, jn) in enumerate(zip(axes.ravel(), names)):
            ax.plot(t, a["targets"][:, j], "k--", lw=1, label="target")
            ax.plot(t, a["measured"][:, j], lw=1.4, label=str(a["sim"]))
            ax.plot(t, b["measured"][:, j], lw=1.4, label=str(b["sim"]))
            ax.set_title(jn, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Joint-target tracking, seeded random targets: {a['sim']} vs {b['sim']}")
        fig.supxlabel("env step")
        fig.supylabel("joint position [rad / m]")
        fig.tight_layout()
        fig.savefig(plot, dpi=110)
        print(f"plot -> {plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--sim", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--steps", type=int, default=120)
    rec.add_argument("--hold", type=int, default=20, help="env steps per random target")
    rec.add_argument("--seed", type=int, default=0)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--plot", default=None)
    args = parser.parse_args()
    if args.cmd == "record":
        record(sim=args.sim, out=args.out, steps=args.steps, seed=args.seed, hold=args.hold)
    else:
        compare(args.a, args.b, plot=args.plot)


if __name__ == "__main__":
    main()
