"""Convert RoboTwin bridge pickles into RoboVerse IL demo dirs (for DP/ACT training).

Each ``collect_bridge.py --rgb`` pickle is one successful bimanual episode that
already carries everything the RoboVerse imitation-learning pipeline needs:

- ``real_vectors`` — RoboTwin's *achieved* 14-D joint state
  ``[L_arm(6), L_grip, R_arm(6), R_grip]`` -> the IL **observation/state**.
- ``vectors``      — the PD *command target*, same 14-D layout -> the IL **action**
  (``joint_qpos_target``).
- ``rgb["head_camera"]`` — per-frame head-camera RGB -> the IL image obs.

This writer emits exactly the on-disk layout ``roboverse_learn/il/data2zarr_dp.py``
consumes -- one ``demo_<id>/`` dir per episode with ``metadata.json`` (``joint_qpos``
= achieved state, ``joint_qpos_target`` = command action) + ``rgb.mp4`` (head cam) --
so the *existing*, unmodified ``data2zarr_dp.py`` then builds the zarr. No change to
the IL trainer; RoboTwin just becomes another data source for it.

Run (roboverse env)::

    python tools/robotwin_integration/robotwin_to_demo.py \\
        --bridge '~/projects/robotwin/data/_rv_bridge/move_can_pot_*.pkl' \\
        --out data/robotwin_demo/move_can_pot --camera head_camera

    # then the standard RoboVerse step:
    python roboverse_learn/il/data2zarr_dp.py --task_name move_can_pot \\
        --expert_data_num <N> --metadata_dir data/robotwin_demo/move_can_pot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import imageio.v2 as imageio
import numpy as np


def _convert_one(bridge: dict, demo_dir: str, camera: str) -> int:
    """Write one demo_<id>/ dir from a bridge pickle; return the frame count (0 if skipped)."""
    vectors = np.asarray(bridge.get("vectors", []), dtype=float)  # command target -> action
    real = np.asarray(bridge.get("real_vectors", []), dtype=float)  # achieved state -> obs
    rgb_all = bridge.get("rgb", {})
    if camera not in rgb_all:
        print(f"  skip {demo_dir}: no '{camera}' RGB (have {list(rgb_all)}); re-collect with --rgb")
        return 0
    rgb = np.asarray(rgb_all[camera], dtype=np.uint8)
    if len(vectors) == 0 or len(real) == 0 or len(rgb) == 0:
        print(f"  skip {demo_dir}: empty vectors/real/rgb ({len(vectors)}/{len(real)}/{len(rgb)})")
        return 0
    # Align lengths: a frame missing its achieved capture would desync state/action/rgb.
    t = min(len(vectors), len(real), len(rgb))
    os.makedirs(demo_dir, exist_ok=True)
    imageio.mimsave(os.path.join(demo_dir, "rgb.mp4"), list(rgb[:t]), fps=20)
    metadata = {
        "joint_qpos": real[:t].tolist(),  # achieved state (obs) -- data2zarr observation_space=joint_pos
        "joint_qpos_target": vectors[:t].tolist(),  # command target (action)
        "task": bridge.get("task"),
        "seed": int(bridge.get("seed", -1)),
        "camera": camera,
        "source": "robotwin_bridge",
    }
    with open(os.path.join(demo_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge", required=True, help="glob of bridge pickles (collect_bridge --rgb output)")
    ap.add_argument("--out", required=True, help="output metadata dir (becomes data2zarr --metadata_dir)")
    ap.add_argument("--camera", default="head_camera", help="which captured camera to use as the image obs")
    ap.add_argument("--start-id", type=int, default=0, help="first demo_<id> index")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.expanduser(args.bridge)))
    if not paths:
        print(f"NO_BRIDGE: no pickles match {args.bridge!r}")
        return 2
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    written = 0
    for path in paths:
        with open(path, "rb") as f:
            bridge = pickle.load(f)
        demo_dir = os.path.join(out, f"demo_{str(args.start_id + written).zfill(4)}")
        t = _convert_one(bridge, demo_dir, args.camera)
        if t:
            print(f"  {os.path.basename(path)} -> {os.path.basename(demo_dir)} ({t} frames)")
            written += 1

    print(f"WROTE {written} demo dir(s) -> {out}  (use --expert_data_num {written} --metadata_dir {out})")
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
