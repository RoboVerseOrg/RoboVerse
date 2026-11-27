"""RoboVerse benchmark suites (``python -m metasim bench suite list``).

A suite is a fixed task list in splits under one protocol (``metasim.bench.suite``). Only tasks
whose demos replay on the suite's backend belong in it: a task the replay baseline cannot solve
would measure the task, not the policy (``python -m metasim bench health`` is the gate).
"""

from __future__ import annotations

from metasim.bench import BenchmarkSuite, EvalProtocol, register_suite

TABLETOP_MUJOCO = register_suite(
    BenchmarkSuite(
        name="tabletop-mujoco",
        simulator="mujoco",
        description=(
            "Single-arm Franka tabletop manipulation on MuJoCo: the ManiSkill-derived tasks whose demos "
            "replay at >= 0.9 on MuJoCo (bench health, 2026-10-25). Scored task-macro over 50 episodes "
            "each. The episode cap is 3x the 75th-percentile recorded demo length, clipped to the task's "
            "own time-out: it tightens push_cube (216), pull_cube (237) and lift_peg_upright (534); pick_cube "
            "and stack_cube keep their 250-step time-out because their recorded demos run that long. Every "
            "task carries a prompt bank, so --prompt-split heldout measures instruction generalisation."
        ),
        splits={
            "core": ("pick_cube", "stack_cube", "push_cube", "pull_cube", "lift_peg_upright"),
        },
        protocol=EvalProtocol(episodes=50, demo_budget_factor=3.0, demo_budget_quantile=0.75, seed=0),
    )
)

__all__ = ["TABLETOP_MUJOCO"]
