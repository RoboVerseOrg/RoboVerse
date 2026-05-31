"""Single source of truth for the robosuite tasks we integrate and verify.

Each entry maps a native robosuite environment to its RoboVerse/MetaSim task name
and records the metadata the parity harness needs (robot, controller, control
freq, horizon). Keep this list small and curated — it drives every runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobosuiteTask:
    """One robosuite task and how it maps into RoboVerse."""

    env_name: str  # native robosuite env, e.g. "Lift"
    metasim_name: str  # RoboVerse task id, e.g. "robosuite.lift"
    robot: str = "Panda"
    controller: str = "BASIC"  # composite controller (v1.5): BASIC -> per-arm OSC_POSE
    control_freq: int = 20
    horizon: int = 200
    # selected object-state observation keys exposed by the env (for obs parity)
    object_obs_keys: tuple[str, ...] = ()
    description: str = ""

    @property
    def slug(self) -> str:
        return self.metasim_name.replace(".", "_")


# Curated benchmark set. Lift/Stack/Door are the canonical single-arm tasks;
# PickPlaceCan + NutAssemblySquare are the standard robomimic policy benchmarks.
ALL_TASKS: list[RobosuiteTask] = [
    RobosuiteTask(
        env_name="Lift",
        metasim_name="robosuite.lift",
        horizon=200,
        object_obs_keys=("cube_pos", "cube_quat", "gripper_to_cube_pos"),
        description="Lift a 0.04 m cube off the table with a Panda + parallel-jaw gripper.",
    ),
    RobosuiteTask(
        env_name="Stack",
        metasim_name="robosuite.stack",
        horizon=200,
        object_obs_keys=("cubeA_pos", "cubeA_quat", "cubeB_pos", "cubeB_quat"),
        description="Stack cube A on top of cube B.",
    ),
    RobosuiteTask(
        env_name="Door",
        metasim_name="robosuite.door",
        horizon=300,
        object_obs_keys=("door_pos", "handle_pos", "hinge_qpos", "handle_qpos"),
        description="Turn the handle and open the door.",
    ),
    RobosuiteTask(
        env_name="PickPlaceCan",
        metasim_name="robosuite.pick_place_can",
        horizon=400,
        object_obs_keys=("Can_pos", "Can_quat"),
        description="Pick the can and place it into its bin (robomimic benchmark task).",
    ),
    RobosuiteTask(
        env_name="NutAssemblySquare",
        metasim_name="robosuite.nut_assembly_square",
        horizon=400,
        object_obs_keys=("SquareNut_pos", "SquareNut_quat"),
        description="Fit the square nut onto the square peg (robomimic benchmark task).",
    ),
]

BY_ENV: dict[str, RobosuiteTask] = {t.env_name: t for t in ALL_TASKS}
BY_METASIM: dict[str, RobosuiteTask] = {t.metasim_name: t for t in ALL_TASKS}


def get(name: str) -> RobosuiteTask:
    """Look up a task by either its robosuite env name or its metasim name."""
    if name in BY_ENV:
        return BY_ENV[name]
    if name in BY_METASIM:
        return BY_METASIM[name]
    raise KeyError(f"unknown robosuite task: {name!r}; known: {[t.env_name for t in ALL_TASKS]}")
