"""Get started: run native LIBERO tasks through RoboVerse (1:1 passthrough).

LIBERO's 130 tasks (suites: spatial / object / goal / 10 / 90) run via their
*original* robosuite + MuJoCo simulator, exposed through RoboVerse. Because the
underlying env is LIBERO's own ``OffScreenRenderEnv``, behaviour is **1:1 by
construction** — no cross-sim porting.

Prerequisites (LIBERO pins conflict with the default RoboVerse env, so use a
dedicated conda env)::

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO && pip install -e .          # robosuite==1.4.0, bddl, robomimic, ...
    pip install gymnasium numpy-quaternion loguru rootutils

Run headless::

    MUJOCO_GL=egl python get_started/libero_passthrough.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen render

from loguru import logger as log

# Importing the package auto-registers every ``Libero/<suite>__<task>`` id.
import roboverse_pack.tasks.libero  # noqa: F401
from roboverse_pack.tasks._passthrough_index import list_passthrough_envs
from roboverse_pack.tasks.libero._passthrough import make_libero_env


def main() -> None:
    # 1) Discover registered LIBERO tasks.
    ids = list_passthrough_envs().get("Libero", [])
    log.info(f"Registered LIBERO tasks: {len(ids)}")
    for env_id in ids[:3]:
        log.info(f"  e.g. {env_id}")

    # 2) Build one task via the native LIBERO path (1:1). ``make_libero_env``
    #    resolves the BDDL, builds the offscreen env, seeds, resets, and applies
    #    init state 0 — the exact construction LIBERO's own eval uses.
    env = make_libero_env("libero_spatial", 0, seed=0, init_state_index=0)

    # 3) Step a few zero actions. NOTE: LIBERO uses the legacy gym 4-tuple API:
    #    step -> (obs, reward, done, info); reset -> obs. ``make_libero_env``
    #    already reset + applied init state 0, so step straight away.
    for t in range(5):
        obs, reward, done, info = env.step([0.0] * 7)
        if t == 0:
            log.info(f"obs keys ({len(obs)}): {sorted(obs)[:6]} ...")
            log.info(f"agentview_image: {obs['agentview_image'].shape} {obs['agentview_image'].dtype}")
            log.info(f"robot0_eef_pos: {obs['robot0_eef_pos']}")
        log.info(f"  step {t}: reward={reward:.3f} done={done}")
        if done:
            break

    env.close()
    log.info("done.")


if __name__ == "__main__":
    main()
