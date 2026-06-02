"""Vendor every RoboTwin asset + trajectory the RoboVerse replay needs into roboverse_data/.

Goal: make ``~/projects/robotwin`` **deletable**. After this runs, the kinematic replay /
side-by-side / object-parity pipeline resolves all assets from ``roboverse_data/robotwin/``
(via :mod:`roboverse_pack.tasks.robotwin._locator`) with no RoboTwin checkout present.

What it migrates (only the SUBSET the 50 bridges actually reference -- ~1.5 GB, not the full
4.4 GB asset zoo):
  * object **visual dirs** for every ``object_meshes[*].visual`` (the GLB/OBJ + its textures);
  * object **collision dirs** when present alongside (physics meshes the URDF ``<collision>`` uses);
  * **URDF instance dirs** for every ``object_meshes[*].urdf`` (``mobility.urdf`` + ``model_data*.json``
    + referenced meshes);
  * the **ALOHA-AgileX embodiment** (``assets/embodiments/aloha-agilex/``: urdf + meshes + srdf);
  * **slim trajectories** -- each bridge with per-frame RGB stripped (RGB is only for DP training
    and is re-renderable from the replay), written to ``roboverse_data/robotwin/bridges/<task>.pkl``.

It also writes ``roboverse_data/robotwin/manifest.json`` (counts + byte totals + per-task object
inventory) so the migration is auditable and the HF upload is reproducible.

Run in the ``roveverse`` env (only needs numpy + stdlib)::

    python tools/robotwin_integration/migrate_assets.py \
        --bridge-dir ~/projects/robotwin/data/_rv_bridge \
        --robotwin-dir ~/projects/robotwin
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil

_RGB_KEYS = ("rgb", "image", "depth", "pointcloud", "point_cloud")
_DEST_PREFIX = os.path.join("roboverse_data", "robotwin")
# RoboTwin-internal relpaths the embodiment lives under (urdf + meshes + srdf; skip the heavy
# curobo *_tmp scratch + collision yml that the replay never reads).
_EMBODIMENT_RELDIR = os.path.join("assets", "embodiments", "aloha-agilex")


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _copy_tree(src: str, dst: str, *, dry: bool) -> int:
    """Copy ``src`` dir to ``dst`` (idempotent: skip files already present + same size)."""
    if not os.path.isdir(src):
        return 0
    copied = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_root = os.path.join(dst, rel) if rel != "." else dst
        if not dry:
            os.makedirs(out_root, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(out_root, f)
            try:
                if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
                    continue
            except OSError:
                pass
            if not dry:
                shutil.copy2(s, d)
            copied += 1
    return copied


def _slim_bridge(d: dict) -> dict:
    """Drop per-frame RGB/depth (re-renderable) -- keep the trajectory + object assets."""
    return {k: v for k, v in d.items() if not any(t in k.lower() for t in _RGB_KEYS)}


def migrate(bridge_dir: str, robotwin_dir: str, *, dry: bool = False, all_instances: bool = True) -> dict:
    bridge_dir = os.path.expanduser(bridge_dir)
    robotwin_dir = os.path.expanduser(robotwin_dir)
    dest = _DEST_PREFIX

    visual_dirs: set[str] = set()  # RoboTwin-internal relpaths of visual dirs
    collision_dirs: set[str] = set()
    urdf_dirs: set[str] = set()
    tasks: list[dict] = []

    pkls = sorted(glob.glob(os.path.join(bridge_dir, "*.pkl")))
    for f in pkls:
        task = os.path.basename(f)[:-4]
        if task.startswith("_") or task == "traj":
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        meshes = d.get("object_meshes") or {}
        tinv = {}
        for name, m in meshes.items():
            if not isinstance(m, dict):
                continue
            tinv[name] = m.get("type", "?")
            if m.get("visual"):
                vdir = os.path.dirname(m["visual"])  # assets/objects/<n>/visual
                visual_dirs.add(vdir)
                # The sibling collision/ dir (physics meshes the URDF <collision> references).
                cdir = os.path.join(os.path.dirname(vdir), "collision")
                if os.path.isdir(os.path.join(robotwin_dir, cdir)):
                    collision_dirs.add(cdir)
            if m.get("urdf"):
                urdf_dirs.add(os.path.dirname(m["urdf"]))
        tasks.append({"task": task, "objects": tinv})

    # URDF objects (laptop/microwave/switch/pot) pick a random INSTANCE per seed via modelid.
    # The captured bridge only references one instance; vendor *every* instance of each referenced
    # URDF category (cheap -- ~80 MB) so replaying any seed of those tasks is also clone-free.
    if all_instances:
        for urel in sorted(urdf_dirs):
            cat = os.path.dirname(urel)  # assets/objects/<category>
            cat_abs = os.path.join(robotwin_dir, cat)
            if not os.path.isdir(cat_abs):
                continue
            for sub in os.listdir(cat_abs):
                inst = os.path.join(cat_abs, sub)
                if sub in ("visual", "collision"):
                    continue
                if os.path.isfile(os.path.join(inst, "mobility.urdf")):
                    urdf_dirs.add(os.path.join(cat, sub))

    # --- copy object assets (visual + collision + urdf instance dirs) ---
    obj_files = 0
    obj_bytes = 0
    for rel in sorted(visual_dirs | collision_dirs | urdf_dirs):
        src = os.path.join(robotwin_dir, rel)
        if not os.path.isdir(src):
            print(f"  [warn] referenced dir missing in clone: {rel}")
            continue
        obj_files += _copy_tree(src, os.path.join(dest, rel), dry=dry)
        obj_bytes += _dir_bytes(src)

    # --- copy the ALOHA-AgileX embodiment ---
    emb_src = os.path.join(robotwin_dir, _EMBODIMENT_RELDIR)
    emb_files = _copy_tree(emb_src, os.path.join(dest, _EMBODIMENT_RELDIR), dry=dry)
    emb_bytes = _dir_bytes(emb_src) if os.path.isdir(emb_src) else 0

    # --- write slim trajectories ---
    traj_dir = os.path.join(dest, "bridges")
    if not dry:
        os.makedirs(traj_dir, exist_ok=True)
    traj_bytes = 0
    n_traj = 0
    for f in pkls:
        task = os.path.basename(f)[:-4]
        if task.startswith("_") or task == "traj":
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        slim = _slim_bridge(d)
        out = os.path.join(traj_dir, f"{task}.pkl")
        if not dry:
            with open(out, "wb") as fh:
                pickle.dump(slim, fh)
            traj_bytes += os.path.getsize(out)
        n_traj += 1

    manifest = {
        "source": "RoboTwin",
        "vendored_into": dest,
        "object_visual_dirs": len(visual_dirs),
        "object_collision_dirs": len(collision_dirs),
        "object_urdf_dirs": len(urdf_dirs),
        "object_files_copied": obj_files,
        "object_bytes": obj_bytes,
        "embodiment_files_copied": emb_files,
        "embodiment_bytes": emb_bytes,
        "trajectories": n_traj,
        "trajectory_bytes": traj_bytes,
        "tasks": tasks,
    }
    if not dry:
        with open(os.path.join(dest, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-dir", default=os.path.expanduser("~/projects/robotwin/data/_rv_bridge"))
    ap.add_argument("--robotwin-dir", default=os.path.expanduser("~/projects/robotwin"))
    ap.add_argument("--dry-run", action="store_true", help="report what would copy without writing")
    ap.add_argument(
        "--no-all-instances",
        dest="all_instances",
        action="store_false",
        help="vendor only the captured instance per URDF category (default: ALL instances, so any "
        "seed of laptop/microwave/switch/pot tasks replays clone-free)",
    )
    args = ap.parse_args()

    m = migrate(args.bridge_dir, args.robotwin_dir, dry=args.dry_run, all_instances=args.all_instances)
    gb = 1e9
    print("\n=== RoboTwin -> roboverse_data migration ===")
    print(
        f"object dirs: {m['object_visual_dirs']} visual + {m['object_collision_dirs']} collision "
        f"+ {m['object_urdf_dirs']} urdf"
    )
    print(f"object files: {m['object_files_copied']} ({m['object_bytes'] / gb:.2f} GB)")
    print(f"embodiment files: {m['embodiment_files_copied']} ({m['embodiment_bytes'] / gb:.2f} GB)")
    print(f"trajectories: {m['trajectories']} ({m['trajectory_bytes'] / 1e6:.1f} MB, RGB stripped)")
    print(f"total vendored: {(m['object_bytes'] + m['embodiment_bytes'] + m['trajectory_bytes']) / gb:.2f} GB")
    print(f"-> {_DEST_PREFIX}/  (manifest.json written)" if not args.dry_run else "(dry run -- nothing written)")


if __name__ == "__main__":
    main()
