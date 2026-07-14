"""Self-contained end-to-end demo of the unified harness on a real RoboVerse task.

Uses the in-process, embodiment-agnostic :class:`~.adapters.scripted.HoldPosePolicy` (echoes
joint-pos obs as the action target) so the whole path — embodiment inference -> typed specs ->
EnvAdapter (tensor action) -> VecEvalRunner (vectorized, per-env episodes) — is exercised with
no checkpoints, servers, or GPU.

    python -m roboverse_learn.eval.harness.demo --task maniskill.pick_cube --num-envs 2
"""

from __future__ import annotations

import argparse

from ._evaluate import evaluate
from .adapters import HoldPosePolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="maniskill.pick_cube")
    ap.add_argument("--simulators", default="mujoco", help="comma-separated for a parity report")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=15)
    args = ap.parse_args()

    sims = args.simulators.split(",")
    result = evaluate(
        args.task,
        HoldPosePolicy(),
        simulators=sims if len(sims) > 1 else sims[0],
        episodes=args.episodes,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
    )
    print("[harness demo]", result)


if __name__ == "__main__":
    main()
