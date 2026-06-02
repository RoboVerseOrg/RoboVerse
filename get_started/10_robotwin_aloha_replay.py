"""Replay a RoboTwin bimanual demonstration in RoboVerse (ALOHA-AgileX).

.. warning::

   **EXPERIMENTAL — not an out-of-the-box path.** Unlike examples 8 and 9, this
   one cannot be run from a clean RoboVerse install. It requires, on the local
   machine:

   - a cloned RoboTwin repo and its ~3.74 GB asset pack (incl. the ALOHA-AgileX
     ``embodiments.zip`` whose URDF this script loads);
   - a separate ``robotwin`` conda env, plus a curobo build for the local GPU
     arch (e.g. sm_120), to *collect* a bridge pickle with
     ``tools/robotwin_integration/collect_bridge.py``;
   - the resulting bridge pickle passed via ``--bridge``.

   It is also **only a partial-fidelity** view: the manipulated object is drawn
   as a primitive-cube proxy (not RoboTwin's real mesh), and only joint *motion*
   has been confirmed to replay -- task *success* has not been verified in
   RoboVerse. Treat this as a data-bridge demo, not a benchmark result.

RoboTwin is a dual-arm manipulation benchmark whose demos are *single-embodiment
bimanual*: one articulation (the ALOHA-AgileX: two arx5 arms on an AgileX base)
whose action spans both arms. RoboVerse's trajectory format expresses this as a
single robot entry keyed by name -- the one-robot case of the same name-keyed
``*_v2`` layout the multi-agent loader uses (see ``8_multiagent_dataset.py`` for
the two-independent-agents case).

This example is the *RoboVerse-side* half of the RoboTwin data bridge:

1.  Load the sim-agnostic trajectory pickle produced (in the ``robotwin`` conda
    env) by ``tools/robotwin_integration/collect_bridge.py`` -- a dense dual-arm
    joint vector per frame plus initial object poses.
2.  Convert it into RoboVerse's name-keyed ``*_v2`` format, mapping RoboTwin's
    14-D ``[L_arm(6), L_grip, R_arm(6), R_grip]`` vector onto the embodiment's
    joints (gripper value in ``[0, 1]`` -> joint via the embodiment's scale).
3.  Load it back through the canonical loader ``get_traj`` and replay it on
    SAPIEN3 to video.

Replay is dof-position-target driven: RoboTwin itself runs on SAPIEN3, so the
same backend family reproduces the recorded joint motion closely (unlike the
cross-simulator ManiSkill case, which uses state replay).

Run (in the ``roboverse`` env, after collecting a bridge pickle)::

    MUJOCO_GL=egl python get_started/10_robotwin_aloha_replay.py \\
        --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl --sim sapien3
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import pickle
from dataclasses import dataclass, field

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
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from roboverse_pack.robots.aloha_agilex_cfg import AlohaAgilexCfg

# Trajectory conversion lives in the robotwin pack so the parity harness and this
# example share one definition of the 14-D vector -> joint mapping.
from roboverse_pack.tasks.robotwin._convert import bridge_to_v2


def _default_bridge() -> str:
    """A live-collection bridge if a RoboTwin clone is present, else the vendored slim one
    (so this example runs after ~/projects/robotwin is deleted)."""
    live = os.path.expanduser("~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl")
    if os.path.exists(live):
        return live
    return os.path.join("roboverse_data", "robotwin", "bridges", "beat_block_hammer.pkl")


@dataclass
class Args:
    """Arguments for the RoboTwin ALOHA-AgileX replay."""

    bridge: str = field(default_factory=_default_bridge)
    sim: str = "sapien3"
    num_envs: int = 1
    save_video: bool = True
    headless: bool = True
    dataset_path: str = "get_started/output/robotwin_aloha_v2.pkl"


def main():
    args = tyro.cli(Args)
    os.makedirs("get_started/output", exist_ok=True)

    if not os.path.exists(args.bridge):
        raise FileNotFoundError(
            f"Bridge pickle not found: {args.bridge}. Collect one first in the robotwin env:\n"
            f"  conda run -n robotwin env MUJOCO_GL=egl python "
            f"tools/robotwin_integration/collect_bridge.py --task beat_block_hammer --out {args.bridge}"
        )
    with open(args.bridge, "rb") as f:
        bridge = pickle.load(f)

    # 1 & 2. Convert the RoboTwin trajectory into a name-keyed v2 dataset.
    bridge_to_v2(bridge, args.dataset_path)

    # 3. Load it back through the canonical loader (single dual-arm embodiment).
    robot = AlohaAgilexCfg()
    if not robot.urdf_path:
        raise FileNotFoundError("ALOHA-AgileX URDF not found; clone RoboTwin and extract embodiments.zip")
    init_states, all_actions, _ = get_traj(args.dataset_path, robot)
    actions = all_actions[0]
    log.info(f"Loaded {len(actions)} action steps driving the dual-arm embodiment.")

    camera = PinholeCameraCfg(
        name="main_camera", pos=[1.3, -0.5, 1.5], look_at=[0.1, -0.2, 0.85], width=640, height=480, data_types=["rgb"]
    )
    objects = [
        # Table top proxy (RoboTwin's table surface sits at z=0.74).
        PrimitiveCubeCfg(name="table", size=(0.8, 0.8, 0.74), color=[0.7, 0.6, 0.5], physics=PhysicStateType.GEOM),
        # The block the demo targets (static red cube, half-size 0.025 in RoboTwin).
        PrimitiveCubeCfg(name="block", size=(0.05, 0.05, 0.05), color=[1.0, 0.0, 0.0], physics=PhysicStateType.XFORM),
    ]
    scenario = ScenarioCfg(
        robots=[robot],
        objects=objects,
        cameras=[camera] if args.save_video else [],
        sim_params=SimParamCfg(dt=0.01),
        decimation=4,
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        add_default_ground=True,
    )

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)
    # Drop the table proxy into the init state at its fixed pose.
    init = init_states[0]
    init["objects"]["table"] = {"pos": [0.0, 0.0, 0.37], "rot": [1.0, 0.0, 0.0, 0.0]}
    handler.set_states([init] * scenario.num_envs)

    obs_saver = None
    if args.save_video:
        video_path = f"get_started/output/10_robotwin_aloha_replay_{args.sim}.mp4"
        obs_saver = ObsSaver(video_path=video_path)
        obs_saver.add(handler.get_states(mode="tensor"))

    # 4. Replay the dual-arm joint trajectory via dof position targets.
    for step, action in enumerate(actions):
        log.debug(f"Step {step}")
        handler.set_dof_targets([action] * scenario.num_envs)
        handler.simulate()
        if obs_saver is not None:
            obs_saver.add(handler.get_states(mode="tensor"))

    if obs_saver is not None:
        obs_saver.save()
        log.info(f"Video saved to get_started/output/10_robotwin_aloha_replay_{args.sim}.mp4")

    handler.close()


if __name__ == "__main__":
    main()
