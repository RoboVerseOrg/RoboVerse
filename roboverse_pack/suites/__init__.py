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
            "each, with the episode cap set to 3x the 75th-percentile demo length so a slow but correct "
            "policy is not cut off and long tasks do not inflate short ones."
        ),
        splits={
            "core": ("pick_cube", "stack_cube", "push_cube", "pull_cube", "lift_peg_upright"),
        },
        protocol=EvalProtocol(episodes=50, demo_budget_factor=3.0, demo_budget_quantile=0.75, seed=0),
    )
)

__all__ = ["TABLETOP_MUJOCO"]
