"""Load ManiSkill ``TwoRobotStackCube-v1`` demo data as a bimanual trajectory.

``TwoRobotStackCube-v1`` is a *collaborative two-arm* ManiSkill task: a left
Panda picks up the blue cube (``cubeA``) while a right Panda places the green
cube (``cubeB``) on the goal region, then the blue cube is stacked on top. Its
demonstrations therefore carry **two agents' states and actions per step**,
which is exactly the multi-agent layout RoboVerse's loader consumes natively.

This example closes the loop end to end:

1.  Fetch the official ManiSkill demo dataset (an ``.h5`` of recorded episodes,
    keyed by agent: ``panda_wristcam-agent-0`` / ``-agent-1``).
2.  Convert one successful episode into RoboVerse's name-keyed ``*_v2`` format --
    one entry **per agent, keyed by robot name** -- the same on-disk layout a
    single-agent dataset uses, so the multi-agent file is just the two-key case.
3.  Load both agents in lock-step with the canonical loader
    ``metasim.utils.demo_util.get_traj`` -- passing the **list** of robots
    returns each agent's slice merged into one namespaced trajectory.
4.  Replay the recorded states on MuJoCo and render a video.

Replay uses the recorded *states* (kinematic playback) rather than open-loop
action targets: the demos were collected under ManiSkill's SAPIEN
``pd_joint_delta_pos`` controller, and closed-loop contact dynamics do not
transfer across simulators, so state replay is the faithful cross-sim view of
the dataset (the actions are still converted and available via ``all_actions``).

Run::

    MUJOCO_GL=egl python examples/9_maniskill_two_robot_stack_cube.py --sim mujoco
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import subprocess
import sys
from dataclasses import dataclass
from glob import glob

import h5py
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

# Map RoboVerse agent names -> the ManiSkill articulation keys in the .h5 demo.
# agent-0 is the left arm (at y<0), agent-1 the right arm (at y>0).
AGENT_TO_ARTICULATION = {
    "panda_left": "panda_wristcam-agent-0",
    "panda_right": "panda_wristcam-agent-1",
}
# Shared rigid objects recorded in every episode (Panda is the same 9-DOF arm
# as RoboVerse's Franka, so the qpos order below matches FrankaCfg's actuators).
OBJECT_NAMES = ["cubeA", "cubeB"]
PANDA_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# ManiSkill packs an articulation env_state as (see mani_skill/utils/structs/
# articulation.py::Articulation.get_state):
#     torch.hstack([pose.p(3), pose.q(4), lin_vel(3), ang_vel(3), qpos, qvel])
# so qpos starts at index 13 (matching set_state's `state[:, 13:13+max_dof]`).
# An actor row is [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13. Quaternions are
# SAPIEN poses, which are scalar-first **wxyz** (identity Pose.q == [1, 0, 0, 0]),
# matching RoboVerse's convention -- so pos = row[:3], rot(wxyz) = row[3:7].
_POSE_QUAT_SLICE = slice(3, 7)
_QPOS_SLICE = slice(13, 13 + len(PANDA_JOINTS))
# Minimum articulation row width we read (through the end of qpos).
_MIN_ART_WIDTH = _QPOS_SLICE.stop


def _validate_schema(traj) -> None:
    """Fail loud if the recorded ManiSkill layout differs from what we slice.

    Guards the hardcoded agent keys and qpos slice so a schema change raises a
    clear error instead of silently reading the wrong columns as qpos.

    Args:
        traj: One episode group from the demo ``.h5``.

    Raises:
        KeyError: If an expected articulation/actor key is missing.
        ValueError: If an articulation row is narrower than ``_MIN_ART_WIDTH``.
    """
    for articulation in AGENT_TO_ARTICULATION.values():
        key = f"env_states/articulations/{articulation}"
        if key not in traj:
            available = list(traj["env_states/articulations"].keys())
            raise KeyError(f"Expected articulation '{key}' not in demo; found articulations: {available}")
        width = traj[key].shape[1]
        if width < _MIN_ART_WIDTH:
            raise ValueError(
                f"Articulation '{articulation}' row width {width} < {_MIN_ART_WIDTH}; the ManiSkill state "
                f"layout changed and qpos slice {_QPOS_SLICE} would read wrong columns."
            )
    for name in OBJECT_NAMES:
        key = f"env_states/actors/{name}"
        if key not in traj:
            available = list(traj["env_states/actors"].keys())
            raise KeyError(f"Expected actor '{key}' not in demo; found actors: {available}")


def _ensure_demo(h5_glob: str, out_dir: str) -> str:
    """Return the demo ``.h5`` path, downloading the dataset if it is absent."""
    hits = glob(h5_glob)
    if hits:
        return hits[0]
    log.info("ManiSkill demo not found locally; downloading TwoRobotStackCube-v1...")
    subprocess.run(
        [sys.executable, "-m", "mani_skill.utils.download_demo", "TwoRobotStackCube-v1", "-o", out_dir],
        check=True,
    )
    hits = glob(h5_glob)
    if not hits:
        raise FileNotFoundError(f"Download finished but no demo matched {h5_glob}")
    return hits[0]


def _agent_slice(traj, articulation: str, agent_name: str) -> dict:
    """Build one agent's name-keyed v2 demo (init_state + per-step states/actions)."""
    art = traj[f"env_states/articulations/{articulation}"][:]
    objs = {name: traj[f"env_states/actors/{name}"][:] for name in OBJECT_NAMES}
    n_frames = art.shape[0]

    def frame_state(i: int) -> dict:
        qpos = art[i, _QPOS_SLICE]
        state = {
            agent_name: {
                "pos": art[i, :3].tolist(),
                "rot": art[i, _POSE_QUAT_SLICE].tolist(),  # wxyz
                "dof_pos": dict(zip(PANDA_JOINTS, qpos.tolist(), strict=False)),
            }
        }
        for name, rows in objs.items():
            state[name] = {"pos": rows[i, :3].tolist(), "rot": rows[i, _POSE_QUAT_SLICE].tolist()}
        return state

    states = [frame_state(i) for i in range(n_frames)]
    # Action target at step t is the next recorded joint pose (state replay's
    # action-space twin); the loader merges these into per-step {robot: action}.
    actions = [
        {"dof_pos_target": dict(zip(PANDA_JOINTS, art[i, _QPOS_SLICE].tolist(), strict=False))}
        for i in range(1, n_frames)
    ]
    return {"init_state": states[0], "actions": actions, "states": states}


def convert_demo_to_v2(h5_path: str, out_path: str, demo_index: int | None) -> int:
    """Convert one ManiSkill episode into a name-keyed multi-agent ``*_v2`` file.

    Picks the first successful episode when ``demo_index`` is ``None``. Returns
    the number of frames in the converted episode.
    """
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if demo_index is not None:
            key = f"traj_{demo_index}"
            if key not in f:
                raise KeyError(f"{key} not in demo (have {len(keys)} episodes)")
        else:
            key = next((k for k in keys if bool(f[k]["success"][-1])), keys[0])
        traj = f[key]
        _validate_schema(traj)
        dataset = {
            name: [_agent_slice(traj, articulation, name)] for name, articulation in AGENT_TO_ARTICULATION.items()
        }
        n_frames = traj["env_states/articulations/panda_wristcam-agent-0"].shape[0]
    dataset["metadata"] = {
        "num_agents": len(AGENT_TO_ARTICULATION),
        "agents": list(AGENT_TO_ARTICULATION),
        "source": "ManiSkill TwoRobotStackCube-v1",
        "source_episode": key,
    }
    save_traj_file(dataset, out_path)
    log.info(f"Converted ManiSkill episode {key} ({n_frames} frames) -> {out_path}")
    return n_frames


@dataclass
class Args:
    """Arguments for the ManiSkill bimanual dataset demo."""

    sim: str = "mujoco"
    num_envs: int = 1
    save_video: bool = True
    headless: bool = True
    demo_index: int | None = None  # None -> first successful episode
    demo_dir: str = "examples/output/ms_demos"
    dataset_path: str = "examples/output/two_robot_stack_cube_v2.pkl"


def main():
    args = tyro.cli(Args)
    os.makedirs("examples/output", exist_ok=True)

    # 1 & 2. Fetch the ManiSkill demos and convert one episode to name-keyed v2.
    h5_glob = os.path.join(args.demo_dir, "TwoRobotStackCube-v1", "**", "trajectory*.h5")
    h5_path = _ensure_demo(h5_glob, args.demo_dir)
    convert_demo_to_v2(h5_path, args.dataset_path, args.demo_index)

    # One RobotCfg per agent: Panda == RoboVerse's Franka (same 9-DOF layout).
    franka = FrankaCfg()
    robots = [franka.replace(name=name) for name in AGENT_TO_ARTICULATION]

    # 3. Load both agents via the canonical loader. Passing the *list* of robots
    #    returns the merged, per-agent-namespaced trajectory.
    init_states, all_actions, all_states = get_traj(args.dataset_path, robots)
    if all_states is None:
        raise RuntimeError("Converted dataset has no states to replay")
    states, actions = all_states[0], all_actions[0]
    log.info(f"Loaded {len(states)} state frames; each action step drives {len(actions[0])} agents: {list(actions[0])}")

    camera = PinholeCameraCfg(
        name="main_camera", pos=[0.9, 0.0, 0.7], look_at=[0.0, 0.0, 0.1], width=640, height=480, data_types=["rgb"]
    )
    scenario = ScenarioCfg(
        robots=robots,
        objects=[
            PrimitiveCubeCfg(
                name="cubeA", size=(0.04, 0.04, 0.04), color=[0.05, 0.16, 0.63], physics=PhysicStateType.RIGIDBODY
            ),
            PrimitiveCubeCfg(
                name="cubeB", size=(0.04, 0.04, 0.04), color=[0.0, 1.0, 0.0], physics=PhysicStateType.RIGIDBODY
            ),
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
        video_path = f"examples/output/9_maniskill_two_robot_stack_cube_{args.sim}.mp4"
        obs_saver = ObsSaver(video_path=video_path)
        obs_saver.add(handler.get_states(mode="tensor"))

    # 4. Replay the recorded states (kinematic playback) for both arms at once.
    #    Pure state replay: set the recorded state, refresh the render, and read
    #    it back -- NO handler.simulate(), which would advance one physics step
    #    and perturb the just-set pose (this mirrors replay_demo.py's
    #    `object_states` branch: set_states -> refresh_render -> get_states).
    for step, state in enumerate(states):
        log.debug(f"Step {step}")
        handler.set_states([state] * scenario.num_envs)
        handler.refresh_render()
        if obs_saver is not None:
            obs_saver.add(handler.get_states(mode="tensor"))

    if obs_saver is not None:
        obs_saver.save()
        log.info(f"Video saved to examples/output/9_maniskill_two_robot_stack_cube_{args.sim}.mp4")

    handler.close()


if __name__ == "__main__":
    main()
