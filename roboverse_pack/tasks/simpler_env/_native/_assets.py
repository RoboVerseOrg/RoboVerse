"""Asset path resolver for the native SimplerEnv port — no upstream/clone dependency.

Resolves the migrated SimplerEnv assets (robot URDFs, scene GLBs, object meshes, model_db JSONs,
real-world overlay images) that live under ``roboverse_data/assets/simpler_env`` +
``roboverse_data/robots``. Resolution order:
  1. ``$ROBOVERSE_DATA`` env var (a roboverse_data checkout),
  2. ``roboverse_data`` next to this repo (default dev layout),
  3. (future) HF-backed download, mirroring the robotwin/mjlab ``_locator`` pattern.

These assets are vendored copies (HF-backed at release); the native port reads ONLY from here,
never from a cloned ``mani_skill2_real2sim`` / ``simpler_env`` tree.
"""

from __future__ import annotations

import os
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[4]  # .../RoboVerse-simpler


def data_root() -> pathlib.Path:
    cands = []
    if os.environ.get("ROBOVERSE_DATA"):
        cands.append(pathlib.Path(os.environ["ROBOVERSE_DATA"]))
    cands.append(_REPO / "roboverse_data")
    for c in cands:
        if c and (c / "assets" / "simpler_env").exists():
            return c
    raise FileNotFoundError(
        "roboverse_data/assets/simpler_env not found. Set $ROBOVERSE_DATA to a roboverse_data "
        "checkout containing the migrated SimplerEnv assets (robots/, assets/simpler_env/)."
    )


def paths() -> dict:
    """Return the canonical native-asset paths as strings."""
    r = data_root()
    a = r / "assets" / "simpler_env"
    return {
        "urdf_google": str(r / "robots/google_robot/urdf/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"),
        "urdf_widowx": str(r / "robots/widowx/widowx_description/wx250s.urdf"),
        "scenes_dir": str(a / "scenes"),
        "models_dir": str(a / "models"),
        "model_db_dir": str(a / "model_db"),
        "overlay_dir": str(a / "real_inpainting"),
        "coke_scene": str(a / "scenes/google_pick_coke_can_1_v4.glb"),
        "coke_overlay": str(a / "scenes/google_coke_can_real_eval_1.png"),
        "coke_collision": str(a / "opened_coke_can/collision.obj"),
        "coke_visual": str(a / "opened_coke_can/textured.dae"),
        "cabinet_urdf": str(a / "cabinet/mk_station.urdf"),
    }
