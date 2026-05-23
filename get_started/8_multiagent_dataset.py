"""Build, save, load and replay a *multi-agent* (bimanual) trajectory dataset.

This example demonstrates that RoboVerse's data model is already multi-agent
native: a single trajectory file stores one entry **per agent, keyed by robot
name** (the same on-disk layout single-agent datasets already use), and actions
are namespaced per agent as ``{robot_name: {"dof_pos_target": {...}}}``.

The flow is:

1.  Procedurally build a coordinated two-arm trajectory.
2.  Save it as a real ``*_v2.pkl`` dataset, keyed by robot name -- exactly the
    format ``metasim.utils.demo_util`` already reads. A single-agent file is
    just the one-key special case, so backwards compatibility is free.
3.  Load it back and zip the per-agent action streams into the per-step,
    per-agent action dict the handler consumes (the small multi-agent loader
    that is the only piece not yet exposed by ``get_traj``).
4.  Replay both arms simultaneously and render a video.

Run::

    python get_started/8_multiagent_dataset.py --sim mujoco
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
from dataclasses import dataclass

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.demo_util.loader import load_traj_file, save_traj_file
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from roboverse_pack.robots import FrankaCfg

# The two agents of the bimanual cell. Any number of agents works the same way.
AGENT_NAMES = ["franka_left", "franka_right"]

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


def _agent_init_state(name: str, base_y: float) -> dict:
    """Per-agent init entry: base pose + home joints, mirrored across the table."""
    return {
        name: {
            "pos": [0.0, base_y, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": dict(FRANKA_HOME),
        },
        # Shared object the two arms coordinate around (a handover cube).
        "cube": {"pos": [0.45, 0.0, 0.05], "rot": [1.0, 0.0, 0.0, 0.0]},
    }


def _scripted_action(name: str, phase: float) -> dict:
    """A smooth coordinated reach: both arms sweep toward the center cube.

    ``franka_right`` mirrors ``franka_left`` on joint 1 so the two arms move as a
    coordinated pair rather than independently -- the visual signature of a
    bimanual task.
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


def build_multiagent_dataset(num_steps: int) -> dict:
    """Build a format-faithful multi-agent dataset: ``{robot_name: [demo, ...]}``."""
    dataset: dict = {}
    base_y = {"franka_left": 0.45, "franka_right": -0.45}
    for name in AGENT_NAMES:
        actions = []
        for t in range(num_steps):
            phase = 2.0 * t / max(num_steps - 1, 1)  # 0 -> 2
            actions.append(_scripted_action(name, phase))
        dataset[name] = [
            {
                "init_state": _agent_init_state(name, base_y[name]),
                "actions": actions,
                "states": None,
            }
        ]
    dataset["metadata"] = {"num_agents": len(AGENT_NAMES), "agents": list(AGENT_NAMES)}
    return dataset


def load_multiagent_traj(path: str, agent_names: list[str], demo_idx: int = 0):
    """Load every agent's slice from one file and zip them in lock-step.

    Returns ``(init_state, combined_actions)`` where ``init_state`` is the
    handler ``{"robots": ..., "objects": ...}`` dict and ``combined_actions`` is a
    list of per-step, per-agent action dicts.

    This is the small multi-agent helper that ``get_traj`` (single-robot) does
    not yet expose; it is the natural place to add native support.
    """
    raw = load_traj_file(path)

    merged_init: dict = {"robots": {}, "objects": {}}
    per_agent_actions = {}
    for name in agent_names:
        demo = raw[name][demo_idx]
        per_agent_actions[name] = demo["actions"]
        for entity, val in demo["init_state"].items():
            entry = {
                "pos": torch.tensor(val["pos"]),
                "rot": torch.tensor(val["rot"]),
                **({"dof_pos": val["dof_pos"]} if "dof_pos" in val else {}),
            }
            if entity == name:
                merged_init["robots"][entity] = entry
            else:  # shared object, only add once
                merged_init["objects"].setdefault(entity, entry)

    horizon = min(len(per_agent_actions[name]) for name in agent_names)
    combined_actions = [{name: per_agent_actions[name][t] for name in agent_names} for t in range(horizon)]
    return merged_init, combined_actions


@dataclass
class Args:
    """Arguments for the multi-agent dataset demo."""

    sim: str = "mujoco"
    num_steps: int = 120
    num_envs: int = 1
    save_video: bool = True
    headless: bool = True
    dataset_path: str = "get_started/output/bimanual_handover_v2.pkl"


def main():
    args = tyro.cli(Args)
    os.makedirs("get_started/output", exist_ok=True)

    # 1 & 2. Build the multi-agent dataset and save it as a real *_v2.pkl file.
    dataset = build_multiagent_dataset(args.num_steps)
    save_traj_file(dataset, args.dataset_path)
    log.info(f"Saved multi-agent dataset (agents={AGENT_NAMES}) -> {args.dataset_path}")

    # 3. Load it back via the multi-agent loader.
    init_state, combined_actions = load_multiagent_traj(args.dataset_path, AGENT_NAMES)
    log.info(f"Loaded {len(combined_actions)} steps; each step drives {len(AGENT_NAMES)} agents.")

    # Build a scene with one robot per agent (same FrankaCfg, distinct names).
    franka = FrankaCfg()
    camera = PinholeCameraCfg(
        name="main_camera", pos=[1.8, 0.0, 1.3], look_at=[0.35, 0.0, 0.3], width=640, height=480, data_types=["rgb"]
    )
    scenario = ScenarioCfg(
        robots=[franka.replace(name=name) for name in AGENT_NAMES],
        objects=[
            PrimitiveCubeCfg(
                name="cube", size=(0.05, 0.05, 0.05), color=[1.0, 0.2, 0.2], physics=PhysicStateType.RIGIDBODY
            )
        ],
        cameras=[camera] if args.save_video else [],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)
    handler.set_states([init_state] * scenario.num_envs)

    obs_saver = None
    if args.save_video:
        video_path = f"get_started/output/8_multiagent_dataset_{args.sim}.mp4"
        obs_saver = ObsSaver(video_path=video_path)
        obs_saver.add(handler.get_states(mode="tensor"))

    # 4. Replay the dataset: every step feeds a per-agent action dict to all envs.
    for step, action in enumerate(combined_actions):
        log.debug(f"Step {step}")
        handler.set_dof_targets([action] * scenario.num_envs)
        handler.simulate()
        if obs_saver is not None:
            obs_saver.add(handler.get_states(mode="tensor"))

    if obs_saver is not None:
        obs_saver.save()
        log.info(f"Video saved to get_started/output/8_multiagent_dataset_{args.sim}.mp4")

    handler.close()


if __name__ == "__main__":
    main()
