"""Cross-sim joint-target tracking parity: SuperDex vs MuJoCo (or any two backends).

Runs the ``examples/1_control_robot.py`` scene with a *seeded* random joint-target sequence on one
backend and records, per env step, the commanded targets and the measured joint positions. Because
SuperDex ships Python 3.12-only wheels while the MuJoCo env here is 3.11, each backend is recorded in
its own environment and the two recordings are compared afterwards::

    python tools/parity/parity_superdex_tracking.py record --sim mujoco   --out /tmp/parity/mujoco.npz
    python tools/parity/parity_superdex_tracking.py record --sim superdex --out /tmp/parity/superdex.npz
    python tools/parity/parity_superdex_tracking.py compare /tmp/parity/mujoco.npz /tmp/parity/superdex.npz --plot /tmp/parity/tracking.png

``compare`` prints, per backend, the mean / max absolute tracking error |q - q_target| over the run
and the mean absolute difference between the two backends' joint trajectories. It does not claim
parity: it measures it (see AGENTS.md, "Parity Is Load-Bearing").

``record --mode drop`` records rigid-object dynamics instead of robot tracking: the cube, sphere and
bbq-sauce bottle of the ``0_static_scene`` tutorial are released from the same poses and their root
positions are logged per env step; ``compare`` then reports the per-object position difference
(settling height, slide/roll distance) between the two backends.
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


def _drop_scenario(sim: str):
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
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
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
        ],
    )


_DROP_INIT = {
    "cube": ([0.3, -0.2, 0.30], [0.9239, 0.0, 0.3827, 0.0]),  # tilted 45 deg about y: lands on an edge, tips over
    "sphere": ([0.4, -0.6, 0.40], [1.0, 0.0, 0.0, 0.0]),
    "bbq_sauce": ([0.7, -0.3, 0.35], [0.7071, 0.7071, 0.0, 0.0]),  # lying on its side: rolls
}


def record_drop(*, sim: str, out: str, steps: int) -> None:
    """Release the three tutorial objects from fixed poses and log their root positions per env step."""
    from metasim.utils.setup_util import get_handler

    scenario = _drop_scenario(sim)
    handler = get_handler(scenario)
    robot = scenario.robots[0]
    init = {
        "objects": {name: {"pos": torch.tensor(p), "rot": torch.tensor(q)} for name, (p, q) in _DROP_INIT.items()},
        "robots": {
            robot.name: {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": dict(robot.default_joint_positions),
            }
        },
    }
    handler.set_states([init])
    names = list(_DROP_INIT)
    traj = []
    for _ in range(steps):
        handler.simulate()
        objs = handler.get_states(mode="dict")[0]["objects"]
        traj.append([np.asarray(objs[n]["pos"], dtype=np.float64) for n in names])
    handler.close()
    out = out if out.endswith(".npz") else out + ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, sim=sim, mode="drop", object_names=np.array(names), positions=np.array(traj))
    final = np.array(traj)[-1]
    print(
        f"[{sim}] drop {steps} steps: final z "
        + ", ".join(f"{n}={final[i][2]:.4f}" for i, n in enumerate(names))
        + f" -> {out}"
    )


def compare_drop(a, b, *, plot: str | None = None) -> None:
    names = list(a["object_names"])
    if names != list(b["object_names"]):
        raise ValueError(f"object order differs: {names} vs {list(b['object_names'])}")
    pa, pb = a["positions"], b["positions"]  # (T, n_obj, 3)
    n = min(len(pa), len(pb))
    for i, name in enumerate(names):
        diff = np.linalg.norm(pa[:n, i] - pb[:n, i], axis=1)
        print(
            f"  {name:>10}: final pos {a['sim']}={np.round(pa[n - 1, i], 4).tolist()} {b['sim']}={np.round(pb[n - 1, i], 4).tolist()}"
            f" | mean|Δpos|={diff.mean() * 1000:.1f} mm  max={diff.max() * 1000:.1f} mm  final={diff[-1] * 1000:.1f} mm"
        )
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(names), 3, figsize=(13, 3 * len(names)), sharex=True, squeeze=False)
        t = np.arange(n)
        for i, name in enumerate(names):
            for k, axis in enumerate("xyz"):
                ax = axes[i, k]
                ax.plot(t, pa[:n, i, k], lw=1.4, label=str(a["sim"]))
                ax.plot(t, pb[:n, i, k], lw=1.4, label=str(b["sim"]))
                ax.set_title(f"{name} {axis} [m]", fontsize=9)
                ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Rigid-object drop dynamics: {a['sim']} vs {b['sim']}")
        fig.supxlabel("env step")
        fig.tight_layout()
        fig.savefig(plot, dpi=110)
        print(f"plot -> {plot}")


def _apply_effort_limit(scenario, effort_limit: float | None) -> None:
    """Give every arm actuator the same explicit ``effort_limit_sim`` on every backend.

    Without it each backend clamps with whatever its asset file says (the Franka MJCF has
    ``forcerange="-40 40"``, the URDF ``<limit effort="87">``), which is the dominant source of
    closed-loop divergence — exactly what the MuJoCo backend warns about at launch.
    """
    if effort_limit is None:
        return
    from metasim.utils.setup_util import get_robot

    robot = scenario.robots[0]
    if isinstance(robot, str):
        robot = get_robot(robot)
        scenario.robots = [robot]
    for name, act in robot.actuators.items():
        if "finger" not in name:
            act.effort_limit_sim = float(effort_limit)


def record(*, sim: str, out: str, steps: int, seed: int, hold: int, effort_limit: float | None = None) -> None:
    from metasim.utils.setup_util import get_handler

    scenario = _scenario(sim)
    _apply_effort_limit(scenario, effort_limit)
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
    if "positions" in a.files:
        if "positions" not in b.files:
            raise ValueError("cannot compare a drop recording with a tracking recording")
        compare_drop(a, b, plot=plot)
        return
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
        for j, (ax, jn) in enumerate(zip(axes.ravel(), names, strict=False)):
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
    rec.add_argument("--mode", choices=("tracking", "drop"), default="tracking")
    rec.add_argument(
        "--effort-limit",
        type=float,
        default=None,
        help="explicit effort_limit_sim [N m] for every arm actuator on this backend (see _apply_effort_limit)",
    )
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--plot", default=None)
    args = parser.parse_args()
    if args.cmd == "record" and args.mode == "drop":
        record_drop(sim=args.sim, out=args.out, steps=args.steps)
    elif args.cmd == "record":
        record(
            sim=args.sim, out=args.out, steps=args.steps, seed=args.seed, hold=args.hold, effort_limit=args.effort_limit
        )
    else:
        compare(args.a, args.b, plot=args.plot)


if __name__ == "__main__":
    main()
