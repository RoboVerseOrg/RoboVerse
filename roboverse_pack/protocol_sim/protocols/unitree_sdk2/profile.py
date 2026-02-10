from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitreeRobotProfile:
    """Static description for a Unitree SDK2-compatible robot.

    The key piece is ``motor_names``, which defines the motor index order used by
    Unitree's LowCmd/LowState messages (hardware order), mapped to the simulator's
    joint names.
    """

    robot_name: str
    msg_type: str  # "hg" (G1/H1-2) or "go" (Go2/H1)
    motor_names: list[str]

    # Topics
    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"
    sportstate_topic: str = "rt/sportmodestate"

    # HG-only: used by some controllers during init (mirrors unitree_mujoco behavior).
    mode_machine: int | None = None
