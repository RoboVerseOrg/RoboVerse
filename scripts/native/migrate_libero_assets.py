"""Vendor LIBERO into roboverse_data: deduped assets + portable per-task MJCF + goal.

This is what makes the ``libero`` / ``robosuite`` Python packages **wholesale
deletable**. For every base LIBERO task it:

1. builds the env from its BDDL and takes the live robosuite-compiled MJCF
   (``e.model.get_xml()``, local asset paths, recompiles bitwise — verified:
   nq/nv/ngeom match, checker 0 mismatch, render MAE ~0.1 vs robosuite);
2. maps each ``file="<abs>"`` to a stable two-namespace relpath
   (``robosuite/…`` for the Franka/gripper/arena meshes, ``libero/…`` for the
   chiliocosm objects/textures), copying each **unique** file once into
   ``roboverse_data/libero/assets/`` (193 files / ~265 MB deduped across all 130);
3. rewrites the MJCF ``file=`` to those relpaths and saves the portable task XML +
   the resolved BDDL goal / OSC config (so a task is fully self-contained from
   MJCF + shared assets + goal, with **no libero import**).

Output tree (uploadable to HF ``RoboVerseOrg/roboverse_data``)::

    roboverse_data/libero/
      assets/robosuite/<tail-after models/assets/>   (21 stl, shared Franka/arena)
      assets/libero/<tail-after libero/assets/>       (objects, textures, meshes)
      tasks/<suite>/<task>.xml                        (portable MJCF, relpaths)
      tasks/<suite>/<task>.goal.json                  (goal + OSC cfg)
      manifest.json

Run (liberoplus env):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus PYTHONPATH=. \\
    python -m scripts.native.migrate_libero_assets --out ../roboverse_data/libero
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil

from scripts.native.export_libero_task import _resolve_goal, build_osc_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BDDL_ROOT = os.path.join(ROOT, "third_party", "LIBERO", "libero", "libero", "bddl_files")
DEMO_ROOT = os.path.join(ROOT, "third_party", "libero_datasets")
_FILE_RE = re.compile(r'file="([^"]+)"')


def abs_to_relpath(p: str):
    """Map a local asset abs-path to a stable two-namespace vendored relpath."""
    if "robosuite/models/assets/" in p:
        return "robosuite/" + p.split("robosuite/models/assets/")[1]
    if "/libero/assets/" in p:
        return "libero/" + p.split("/libero/assets/")[1]
    return None


def migrate(out_dir, limit):
    assets_dir = os.path.join(out_dir, "assets")
    tasks_dir = os.path.join(out_dir, "tasks")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)
    from libero.libero.envs import OffScreenRenderEnv

    copied = {}  # relpath -> size (dedup: copy each unique asset once)
    manifest = {"tasks": [], "assets": {}}
    bddls = sorted(glob.glob(os.path.join(BDDL_ROOT, "**", "*.bddl"), recursive=True))
    for k, bddl in enumerate(bddls):
        suite = os.path.basename(os.path.dirname(bddl))
        base = os.path.basename(bddl)[: -len(".bddl")]
        if limit and k >= limit:
            break
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(0)  # LIBERO jitters fixture placement ~5e-3 per reset; seed for a reproducible scene
        env.reset()
        e = env.env
        goal = _resolve_goal(e)
        cfg = build_osc_cfg(e)
        xml = e.model.get_xml()
        # robosuite overrides a few compiled body_pos/body_quat at runtime (fixture
        # placement) that get_xml() doesn't serialize -> capture them so the vendored
        # model can be made bitwise-identical on load.
        m0 = e.sim.model._model
        body_pos = m0.body_pos.tolist()
        body_quat = m0.body_quat.tolist()
        env.close()

        refs = _FILE_RE.findall(xml)
        unresolved = [p for p in refs if abs_to_relpath(p) is None]
        if unresolved:
            raise RuntimeError(f"{suite}/{base}: unresolved asset roots: {unresolved[:3]}")
        for p in set(refs):  # copy each unique asset once (dedup across all tasks)
            rel = abs_to_relpath(p)
            if rel not in copied:
                dst = os.path.join(assets_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(p, dst)
                copied[rel] = os.path.getsize(p)
        xml_portable = _FILE_RE.sub(lambda mm: f'file="{abs_to_relpath(mm.group(1))}"', xml)

        sdir = os.path.join(tasks_dir, suite)
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, base + ".xml"), "w") as f:
            f.write(xml_portable)
        json.dump(
            {"goal": goal, "osc": cfg, "suite": suite, "base": base, "body_pos": body_pos, "body_quat": body_quat},
            open(os.path.join(sdir, base + ".goal.json"), "w"),
            indent=2,
        )
        n_refs = len(_FILE_RE.findall(xml_portable))
        manifest["tasks"].append({"suite": suite, "task": base, "xml_refs": n_refs, "n_goal_preds": len(goal)})
        print(f"[{k + 1:3d}/130] {suite:14s} {base[:44]:44s} refs={n_refs}  uniq_assets={len(copied)}")

    manifest["assets"] = {
        "n_files": len(copied),
        "total_bytes": sum(copied.values()),
        "by_namespace": {
            ns: {
                "n": sum(1 for r in copied if r.startswith(ns + "/")),
                "bytes": sum(s for r, s in copied.items() if r.startswith(ns + "/")),
            }
            for ns in ("robosuite", "libero")
        },
        "sha256": hashlib.sha256("".join(sorted(f"{r}:{s}" for r, s in copied.items())).encode()).hexdigest()[:16],
    }
    manifest["n_tasks"] = len(manifest["tasks"])
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(
        f"\n=== vendored {manifest['n_tasks']} tasks; {len(copied)} unique assets "
        f"({manifest['assets']['total_bytes'] / 1e6:.1f} MB) -> {out_dir} ==="
    )
    print(json.dumps(manifest["assets"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "roboverse_data", "libero"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    migrate(os.path.abspath(args.out), args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
