"""CLI: collect IL demonstrations from a trained RL policy (RL -> IL bridge).

Example (manipulation, checker-based success)::

    python -m roboverse_learn.fusion.collect \
        --task mjlab.lift_cube_yam_v2 --robot yam --sim mujoco \
        --checkpoint outputs/rsl_rl_ppo/.../model_499.pt \
        --out_dir roboverse_demo/lift_cube_yam --num_demos 100

Then the standard IL pipeline consumes the output unchanged::

    python roboverse_learn/il/data2zarr_dp.py --task_name lift_cube_yam \
        --metadata_dir roboverse_demo/lift_cube_yam/success --expert_data_num 100
    python roboverse_learn/il/train.py ...

For continuous locomotion (no success checker) pass ``--keep_all`` so every
completed episode is kept rather than only checker-terminated ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro
from loguru import logger as log


@dataclass
class Args:
    task: str
    """RoboVerse task name (same as RL training, e.g. mjlab.lift_cube_yam_v2)."""
    checkpoint: str
    """Path to a trained policy: TorchScript policy.pt, actor state dict, or rsl-rl model_*.pt."""
    out_dir: str
    """Demos written under <out_dir>/success/demo_XXXX (canonical IL layout)."""
    robot: str = "yam"
    """Robot name (must match the trained task)."""
    sim: str = "mujoco"
    """Simulator backend."""
    num_demos: int = 100
    """Target number of successful demos."""
    num_envs: int = 1
    """Parallel rollout envs."""
    max_steps: int = 2000
    """Safety cap on total rollout steps."""
    device: str = "cuda:0"
    """Inference device."""
    keep_all: bool = False
    """Keep every completed episode (locomotion) instead of only checker-terminated ones."""
    task_desc: str = ""
    """Optional natural-language task description stored in each demo's metadata."""


def main(args: Args) -> None:
    from roboverse_learn.fusion.rl_to_demo import collect_demos_from_policy

    success_fn = (lambda terminated, cache, info: True) if args.keep_all else None
    n = collect_demos_from_policy(
        task=args.task,
        robot=args.robot,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        sim=args.sim,
        num_demos=args.num_demos,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        device=args.device,
        success_fn=success_fn,
        task_desc=args.task_desc,
    )
    log.info(f"[fusion.collect] collected {n} demos under {args.out_dir}/success")


if __name__ == "__main__":
    main(tyro.cli(Args))
