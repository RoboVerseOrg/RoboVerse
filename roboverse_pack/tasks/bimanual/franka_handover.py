"""A runnable multi-agent (bimanual) demo task: two Franka arms, one trajectory.

This is the registered, pipeline-driven counterpart to the standalone example
``examples/8_multiagent_dataset.py``. Because the task scenario declares **two
robots** (``franka_left`` / ``franka_right``), the canonical replay pipeline
``scripts/advanced/replay_demo.py`` passes the full robot list to ``get_traj``
and drives both arms from one trajectory file::

    MUJOCO_GL=egl python scripts/advanced/replay_demo.py --task bimanual.franka_handover --sim mujoco --headless

The trajectory is procedurally generated (a coordinated two-arm reach) and cached
on first use, so the task is runnable out of the box with no external dataset.

The trajectory builder below is the **single source of truth**: the get-started
example imports it instead of redefining it.
"""

from __future__ import annotations

import math
import os
import tempfile

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.demo_util.loader import save_traj_file
from roboverse_pack.robots import FrankaCfg

# The two agents of the bimanual cell. Any number of agents works the same way.
AGENT_NAMES = ["franka_left", "franka_right"]

# Per-agent base pose: the two arms face each other across the table (y axis).
AGENT_BASE_Y = {"franka_left": 0.45, "franka_right": -0.45}

# The shared object the two arms coordinate around (a handover cube).
CUBE_NAME = "cube"
CUBE_POS = [0.45, 0.0, 0.05]

# Franka home pose, shared by both arms.
FRANKA_HOME = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785398,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356194,
    "panda_joint5": 0.0,
    "panda_joint6": 1.570796,
    "panda_joint7": 0.785398,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

# Cached trajectory file. Keep "v2" in the name so get_traj selects the v2 reader.
DATASET_PATH = os.path.join(tempfile.gettempdir(), "roboverse_bimanual_franka_handover_v2.pkl")

DEFAULT_NUM_STEPS = 120


def _agent_init_state(name: str) -> dict:
    """Per-agent init entry: base pose + home joints + the shared cube."""
    return {
        name: {
            "pos": [0.0, AGENT_BASE_Y[name], 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": dict(FRANKA_HOME),
        },
        CUBE_NAME: {"pos": list(CUBE_POS), "rot": [1.0, 0.0, 0.0, 0.0]},
    }


def _scripted_action(name: str, phase: float) -> dict:
    """A smooth coordinated reach: both arms sweep toward the center cube.

    ``franka_right`` mirrors ``franka_left`` on joint 1 so the two arms move as a
    coordinated pair rather than independently -- the visual signature of a
    bimanual task.

    Args:
        name: Agent name (used to mirror the right arm).
        phase: Trajectory phase in ``[0, 2]`` (ease-in-out reach then retract).

    Returns:
        A ``{"dof_pos_target": {...}}`` action dict for this agent.
    """
    s = 0.5 - 0.5 * math.cos(phase * math.pi)  # ease-in-out 0 -> 1 -> 0 over [0, 2]
    mirror = -1.0 if name == "franka_right" else 1.0
    target = dict(FRANKA_HOME)
    target["panda_joint1"] = mirror * 0.8 * s
    target["panda_joint2"] = -0.785398 + 0.5 * s
    target["panda_joint4"] = -2.356194 + 0.6 * s
    # Close the gripper as the arms converge, open as they retract.
    grip = 0.04 - 0.035 * s
    target["panda_finger_joint1"] = grip
    target["panda_finger_joint2"] = grip
    return {"dof_pos_target": target}


def build_multiagent_dataset(num_steps: int = DEFAULT_NUM_STEPS) -> dict:
    """Build a format-faithful multi-agent dataset: ``{robot_name: [demo, ...]}``.

    Args:
        num_steps: Number of trajectory steps per agent.

    Returns:
        A name-keyed ``*_v2`` dataset dict (one entry per agent plus ``metadata``).
    """
    dataset: dict = {}
    for name in AGENT_NAMES:
        actions = []
        for t in range(num_steps):
            phase = 2.0 * t / max(num_steps - 1, 1)  # 0 -> 2
            actions.append(_scripted_action(name, phase))
        dataset[name] = [
            {
                "init_state": _agent_init_state(name),
                "actions": actions,
                "states": None,
            }
        ]
    dataset["metadata"] = {"num_agents": len(AGENT_NAMES), "agents": list(AGENT_NAMES)}
    return dataset


def ensure_handover_dataset(path: str = DATASET_PATH, num_steps: int = DEFAULT_NUM_STEPS) -> str:
    """Write the procedurally-built handover trajectory to ``path``.

    The dataset is deterministic and cheap to build, so it is regenerated every
    time rather than reused on existence -- this avoids serving a stale cached
    file if the builder or ``num_steps`` changes between runs.

    Args:
        path: Destination ``*_v2.pkl`` path.
        num_steps: Number of trajectory steps per agent.

    Returns:
        The dataset path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_traj_file(build_multiagent_dataset(num_steps), path)
    return path


@register_task("bimanual.franka_handover")
class FrankaHandoverTask(BaseTaskEnv):
    """Two coordinated Franka arms replaying a procedurally-built handover reach."""

    scenario = ScenarioCfg(
        robots=[FrankaCfg(name=name) for name in AGENT_NAMES],
        objects=[
            PrimitiveCubeCfg(
                name=CUBE_NAME, size=(0.05, 0.05, 0.05), color=[1.0, 0.2, 0.2], physics=PhysicStateType.RIGIDBODY
            )
        ],
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    max_episode_steps = DEFAULT_NUM_STEPS
    traj_filepath = DATASET_PATH

    def __init__(self, scenario=None, device=None) -> None:
        """Generate the cached handover trajectory before constructing the env."""
        ensure_handover_dataset(self.traj_filepath)
        super().__init__(scenario, device)

    def _get_initial_states(self) -> list[dict]:
        """Both arms at home plus the shared cube, replicated per env."""
        merged_robots, merged_objects = {}, {}
        for name in AGENT_NAMES:
            entry = _agent_init_state(name)
            merged_robots[name] = entry[name]
            merged_objects[CUBE_NAME] = entry[CUBE_NAME]
        return [
            {
                "objects": {k: dict(v) for k, v in merged_objects.items()},
                "robots": {k: {**v, "dof_pos": dict(v["dof_pos"])} for k, v in merged_robots.items()},
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]
