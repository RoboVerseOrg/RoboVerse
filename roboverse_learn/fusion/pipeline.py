"""End-to-end IL+RL fusion pipeline orchestrator.

Chains the *existing* RoboVerse entry points into one command so the full
RL -> demos -> IL loop (and optional IL->RL warm-start) can be run or inspected
as a single unit. It does not reimplement any trainer — each stage shells out to
the script that already owns it:

    rl-train   roboverse_learn/rl/rsl_rl/ppo.py        (train an RL policy)
    collect    roboverse_learn/fusion/collect.py       (RL policy -> demos)
    to-zarr    roboverse_learn/il/data2zarr_dp.py       (demos -> zarr dataset)
    il-train   roboverse_learn/il/train.py             (train an IL policy)

Use ``--dry_run`` to print the exact commands without executing them (handy for
reproducibility and for wiring into a launcher). Select a subset of stages with
``--stages`` (default: collect,to-zarr,il-train — assumes you already have an RL
checkpoint; add ``rl-train`` to train one first).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

import tyro
from loguru import logger as log

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ALL_STAGES = ("rl-train", "collect", "to-zarr", "il-train")


@dataclass
class Args:
    task: str
    """RoboVerse task name (e.g. mjlab.lift_cube_yam_v2)."""
    robot: str = "yam"
    """Robot name."""
    sim: str = "mujoco"
    """Simulator backend."""
    name: str = "fusion_run"
    """Short tag used for demo/zarr/checkpoint dirs."""
    checkpoint: str = ""
    """RL checkpoint for `collect`. Required unless `rl-train` is in --stages
    (then the just-trained policy.pt is used)."""
    model_dir: str = ""
    """Where rl-train writes / collect reads the policy (default outputs/<name>)."""
    demo_dir: str = ""
    """Where collect writes demos (default roboverse_demo/<name>)."""
    num_demos: int = 100
    """Demos to collect."""
    rl_iterations: int = 1000
    """rl-train iterations (only if 'rl-train' in stages)."""
    stages: list[str] = field(default_factory=lambda: ["collect", "to-zarr", "il-train"])
    """Subset/order of {rl-train, collect, to-zarr, il-train}."""
    keep_all: bool = False
    """Pass --keep_all to collect (locomotion / no success checker)."""
    dry_run: bool = False
    """Print the commands instead of running them."""


def build_commands(args: Args) -> list[list[str]]:
    """Return the ordered shell commands for the selected stages.

    Pure (no side effects) so it is unit-testable; :func:`main` runs them.
    """
    bad = [s for s in args.stages if s not in _ALL_STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid: {_ALL_STAGES}")

    model_dir = args.model_dir or os.path.join("outputs", args.name)
    demo_dir = args.demo_dir or os.path.join("roboverse_demo", args.name)
    checkpoint = args.checkpoint or os.path.join(model_dir, "policy.pt")
    py = sys.executable

    cmds: list[list[str]] = []
    for stage in args.stages:
        if stage == "rl-train":
            cmds.append([
                py,
                os.path.join(_REPO, "roboverse_learn", "rl", "rsl_rl", "ppo.py"),
                "--task",
                args.task,
                "--robot",
                args.robot,
                "--sim",
                args.sim,
                "--model_dir",
                model_dir,
                "--max_iterations",
                str(args.rl_iterations),
            ])
        elif stage == "collect":
            cmd = [
                py,
                "-m",
                "roboverse_learn.fusion.collect",
                "--task",
                args.task,
                "--robot",
                args.robot,
                "--sim",
                args.sim,
                "--checkpoint",
                checkpoint,
                "--out_dir",
                demo_dir,
                "--num_demos",
                str(args.num_demos),
            ]
            if args.keep_all:
                cmd.append("--keep_all")
            cmds.append(cmd)
        elif stage == "to-zarr":
            cmds.append([
                py,
                os.path.join(_REPO, "roboverse_learn", "il", "data2zarr_dp.py"),
                "--task_name",
                args.name,
                "--metadata_dir",
                os.path.join(demo_dir, "success"),
                "--expert_data_num",
                str(args.num_demos),
            ])
        elif stage == "il-train":
            cmds.append([
                py,
                os.path.join(_REPO, "roboverse_learn", "il", "train.py"),
                f"task.dataset.zarr_path=data_policy/{args.name}_{args.num_demos}.zarr",
            ])
    return cmds


def main(args: Args) -> None:
    cmds = build_commands(args)
    log.info(f"[fusion.pipeline] {len(cmds)} stage(s): {', '.join(args.stages)}")
    for i, cmd in enumerate(cmds, 1):
        printable = " ".join(cmd)
        log.info(f"[fusion.pipeline] ({i}/{len(cmds)}) {printable}")
        if args.dry_run:
            continue
        res = subprocess.run(cmd, cwd=_REPO, check=False)
        if res.returncode != 0:
            log.error(f"[fusion.pipeline] stage {args.stages[i - 1]!r} failed (exit {res.returncode}); stopping.")
            raise SystemExit(res.returncode)
    log.info("[fusion.pipeline] done" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main(tyro.cli(Args))
