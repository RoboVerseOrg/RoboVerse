"""Vendor LIBERO's mesh/texture assets (deduplicated) into roboverse_data.

The native LIBERO tasks (``roboverse_pack/tasks/libero/native_libero.py``) load each
bundle's MJCF and resolve its ``file=`` mesh/texture refs from the vendored
``roboverse_data/libero/assets/{robosuite,libero}/`` tree (see
``_native_util.roboverse_data_assets``) — so they run with the ``libero`` /
``robosuite`` *packages* uninstalled. This one-time tool builds that tree by walking
every base LIBERO task's compiled scene and copying each **unique** referenced asset
once (the Franka/arena meshes are shared by all 130 tasks, so the tree is ~265 MB
rather than per-task copies). Upload the result to the ``RoboVerseOrg/roboverse_data``
HF dataset under ``libero/assets/``.

Run with LIBERO installed (e.g. the ``liberoplus`` env)::

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus \
        python -m tools.libero_integration.vendor_shared_assets \
        --bddl-root third_party/LIBERO/libero/libero/bddl_files \
        --out roboverse_data/libero/assets
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

_FILE_RE = re.compile(r'file="([^"]+)"')


def _relpath(abs_path: str) -> str | None:
    """Map a local asset abs-path to its stable two-namespace vendored relpath."""
    if "robosuite/models/assets/" in abs_path:
        return "robosuite/" + abs_path.split("robosuite/models/assets/")[1]
    if "/libero/assets/" in abs_path:
        return "libero/" + abs_path.split("/libero/assets/")[1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-root", default="third_party/LIBERO/libero/libero/bddl_files")
    ap.add_argument("--out", default="roboverse_data/libero/assets")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv

    os.makedirs(args.out, exist_ok=True)
    copied: dict[str, int] = {}
    for k, bddl in enumerate(sorted(glob.glob(os.path.join(args.bddl_root, "**", "*.bddl"), recursive=True))):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        xml = env.env.model.get_xml()
        env.close()
        refs = set(_FILE_RE.findall(xml))
        unresolved = [r for r in refs if _relpath(r) is None]
        if unresolved:
            raise RuntimeError(f"{os.path.basename(bddl)}: unresolved asset roots: {unresolved[:3]}")
        for ref in refs:
            rel = _relpath(ref)
            if rel not in copied:
                dst = os.path.join(args.out, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(ref, dst)
                copied[rel] = os.path.getsize(ref)
        print(f"[{k + 1:3d}] {os.path.basename(bddl)[:48]:48s} unique_assets={len(copied)}", flush=True)

    manifest = {
        "n_assets": len(copied),
        "total_bytes": sum(copied.values()),
        "by_namespace": {ns: sum(1 for r in copied if r.startswith(ns + "/")) for ns in ("robosuite", "libero")},
    }
    json.dump(manifest, open(os.path.join(args.out, "..", "assets_manifest.json"), "w"), indent=2)
    print(f"\nvendored {len(copied)} unique assets ({manifest['total_bytes'] / 1e6:.1f} MB) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
