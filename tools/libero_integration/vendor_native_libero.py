"""Build a native-LIBERO task bundle from a demo HDF5 + its BDDL file.

Writes ``roboverse_pack/tasks/libero/native_bundles/<name>/``:
  model.xml  — the demo's embedded robosuite MJCF (raw; remapped at load time)
  goal.json  — parsed BDDL ``:goal`` predicate terms
  init.npz   — a demo's initial qpos/qvel

Run while LIBERO assets are available; the bundle itself needs no libero/robosuite.

    python -m tools.libero_integration.vendor_native_libero \
        --hdf5 datasets/libero/butter_demo.hdf5 \
        --bddl <libero>/bddl_files/libero_object/pick_up_the_butter_and_place_it_in_the_basket.bddl \
        --name butter
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from roboverse_pack.tasks.libero._native_util import parse_goal, remap_libero_model  # noqa: E402

_BUNDLES = Path(__file__).resolve().parents[2] / "roboverse_pack/tasks/libero/native_bundles"
_ASSETS = str(Path(__file__).resolve().parents[2] / "third_party/LIBERO/libero/libero/assets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--demo", default="demo_0")
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    g = f["data"][args.demo]
    model_file = g.attrs["model_file"]
    states = g["states"][()]
    f.close()

    # determine nq/nv via a load of the remapped model
    m = mujoco.MjModel.from_xml_string(remap_libero_model(model_file, _ASSETS))
    nq, nv = m.nq, m.nv
    qpos0 = states[0][1 : 1 + nq]
    qvel0 = states[0][1 + nq : 1 + nq + nv]
    goal = parse_goal(Path(args.bddl).read_text())

    out = _BUNDLES / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.xml").write_text(model_file)  # raw; remapped at load
    (out / "goal.json").write_text(json.dumps(goal))
    np.savez(out / "init.npz", qpos=qpos0, qvel=qvel0)
    print(f"vendored native bundle '{args.name}': nq={nq} nv={nv} goal={goal} -> {out}")


if __name__ == "__main__":
    main()
