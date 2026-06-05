"""Build, save, load and replay a *multi-agent* (bimanual) trajectory dataset.

RoboVerse's data model is multi-agent native: one trajectory file stores one
entry **per agent, keyed by robot name** -- the same on-disk layout single-agent
datasets already use, so a single-agent file is just the one-key special case
(backwards compatible by construction). Actions are namespaced per agent as
``{robot_name: {"dof_pos_target": {...}}}``.

The flow is:

1.  Procedurally build a coordinated two-arm trajectory.
2.  Save it as a real ``*_v2.pkl`` dataset keyed by robot name.
3.  Load every agent's slice in lock-step with the canonical loader
    ``metasim.utils.demo_util.get_traj`` -- passing the **list** of robots
    instead of a single robot returns the per-step, per-agent action dict the
    handler consumes.
4.  Replay both arms simultaneously and render a video.

Run::

    MUJOCO_GL=egl python get_started/8_multiagent_dataset.py --sim mujoco
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass

import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.demo_util.loader import save_traj_file
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from roboverse_pack.robots import FrankaCfg

# The trajectory builder and constants live with the registered task (single
# source of truth). The same dataset is replayable through the canonical pipeline:
#   python scripts/advanced/replay_demo.py --task bimanual.franka_handover --sim mujoco
from roboverse_pack.tasks.bimanual.franka_handover import AGENT_NAMES, CUBE_NAME, build_multiagent_dataset


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

    # Build a scene with one robot per agent (same FrankaCfg, distinct names).
    franka = FrankaCfg()
    robots = [franka.replace(name=name) for name in AGENT_NAMES]

    # 3. Load it back via the canonical loader. Passing the *list* of robots
    #    (instead of a single robot) returns the merged, per-agent-namespaced
    #    trajectory: init_states[d]["robots"] holds every arm, and each action
    #    step is {robot_name: {"dof_pos_target": ...}} for all arms.
    init_states, all_actions, _ = get_traj(args.dataset_path, robots)
    combined_actions = all_actions[0]
    log.info(f"Loaded {len(combined_actions)} steps; each step drives {len(AGENT_NAMES)} agents.")

    camera = PinholeCameraCfg(
        name="main_camera", pos=[1.8, 0.0, 1.3], look_at=[0.35, 0.0, 0.3], width=640, height=480, data_types=["rgb"]
    )
    scenario = ScenarioCfg(
        robots=robots,
        objects=[
            PrimitiveCubeCfg(
                name=CUBE_NAME, size=(0.05, 0.05, 0.05), color=[1.0, 0.2, 0.2], physics=PhysicStateType.RIGIDBODY
            )
        ],
        cameras=[camera] if args.save_video else [],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)
    handler.set_states([init_states[0]] * scenario.num_envs)

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
