"""Generate native LIBERO-plus task bundles (runs in the dedicated robosuite-1.4 env).

LIBERO-plus ships its perturbed scenes as 16k+ BDDL variants plus a single 6.4GB
``assets.zip`` (hundreds of thousands of new-object meshes, light scenes, …) that
does not fit on a normal disk. We don't need all of it: each perturbed task only
references a handful of asset files. So we build each task's robosuite env with a
*lazy* asset layer — on a missing-file error we extract exactly that member from
the remote zip (HTTP range request, no full download), drop it into the overlay
assets dir, and retry. Once the env builds we read its compiled ``model_file``
(self-contained MJCF: perturbed lights/cameras inline, mesh refs to local files)
plus the demo-style initial ``qpos/qvel`` and the parsed BDDL goal, and write a
native bundle (model.xml + goal.json + init.npz) — the same shape the base-LIBERO
native pipeline consumes. The shared MetaSim env then verifies bitwise engine
parity + native BDDL success and registers each as a first-class task.

    LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg PYTHONPATH=third_party/LIBERO-plus \
    MUJOCO_GL=egl /venv/libero_plus/bin/python -m tools.libero_integration.gen_liberoplus \
        --tasks tools/libero_integration/liberoplus_tasks.json --out datasets/liberoplus
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from remotezip import RemoteZip

_ZIP_URL = "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
# overlay assets dir (custom_asset_dir = LIBERO-plus pkg ../assets); lazily filled
_OVERLAY = Path("third_party/LIBERO-plus/libero/libero/assets").resolve()


class LazyZipAssets:
    """Extract individual members of the remote LIBERO-plus assets.zip on demand."""

    def __init__(self, url: str = _ZIP_URL):
        self._z = RemoteZip(url)
        # index members by their path *after* ``/assets/`` (collector prefix varies)
        self._by_rel: dict[str, str] = {}
        for n in self._z.namelist():
            if "/assets/" in n and not n.endswith("/"):
                self._by_rel.setdefault(n.split("/assets/", 1)[1], n)
        self.fetched = 0

    def fetch(self, rel: str) -> bool:
        """Extract ``assets/<rel>`` into the overlay; return True if written/exists."""
        rel = os.path.normpath(rel)  # collapse ``scenes/lights/../textures`` → ``scenes/textures``
        dst = _OVERLAY / rel
        if dst.exists():
            return True
        member = self._by_rel.get(rel)
        if member is None:  # fall back to matching by trailing path / basename
            cand = [m for r, m in self._by_rel.items() if r.endswith(rel) or r.endswith(os.path.basename(rel))]
            member = cand[0] if cand else None
        if member is None:
            return False
        # A new_objects instance (``new_objects/<cat>/<inst>/...``) is a whole MJCF
        # tree (mesh/texture parts) — bulk-extract the instance dir in one pass so we
        # don't thrash the retry loop fetching dozens of sibling meshes one at a time.
        parts = rel.split("/")
        if parts[0] == "new_objects" and len(parts) >= 3:
            prefix = "/".join(parts[:3]) + "/"
            n = 0
            for r, m in self._by_rel.items():
                if r.startswith(prefix):
                    out = _OVERLAY / r
                    if not out.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(self._z.read(m))
                        n += 1
            self.fetched += n
            return n > 0 or dst.exists()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self._z.read(member))
        self.fetched += 1
        return True

    def close(self):
        self._z.close()


def _missing_rel(exc: Exception) -> str | None:
    """Pull the ``assets/<rel>`` path out of a build error (file-not-found / compile)."""
    msg = str(exc)
    m = re.search(r"/assets/([^\s'\"]+\.(?:xml|png|stl|obj|mtl|jpg|jpeg|msh))", msg)
    return m.group(1) if m else None


def build_task(bddl_path: str, assets: LazyZipAssets, max_fetch: int = 64):
    """Build the perturbed robosuite env, lazily fetching missing assets; return state."""
    from libero.libero.envs import OffScreenRenderEnv

    last = None
    for _ in range(max_fetch):
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=84, camera_widths=84)
            env.reset()
            sim = env.sim
            xml = sim.model.get_xml()
            qpos = sim.data.qpos.copy()
            qvel = sim.data.qvel.copy()
            nq, nv = int(sim.model.nq), int(sim.model.nv)
            lang = env.language_instruction
            env.close()
            return {"xml": xml, "qpos": qpos, "qvel": qvel, "nq": nq, "nv": nv, "lang": lang}
        except Exception as e:
            rel = _missing_rel(e)
            if rel is None or not assets.fetch(rel):
                raise
            last = rel
    raise RuntimeError(f"exceeded max_fetch (last missing {last})")


def parse_goal(bddl_text: str):
    m = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl_text, re.S)
    block = m.group(1) if m else bddl_text
    out = []
    for pred, args in re.findall(r"\((\w+)\s+([^()]*?)\)", block):
        if pred.lower() != "and":
            out.append([pred, args.split()])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True, help="JSON list of [dim, suite, bddl_relpath]")
    ap.add_argument("--out", type=Path, required=True, help="output bundle root")
    ap.add_argument("--bddl-root", default="third_party/LIBERO-plus/libero/libero/bddl_files")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())
    assets = LazyZipAssets()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, ok, err = [], 0, 0
    for i, (dim, suite, rel) in enumerate(tasks):
        bddl = os.path.join(args.bddl_root, suite, rel)
        name = f"{dim}__{suite.replace('libero_', '')}__{Path(rel).stem}"[:120]
        try:
            st = build_task(bddl, assets)
            goal = parse_goal(Path(bddl).read_text())
            bdir = args.out / name
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / "model.xml").write_text(st["xml"])
            (bdir / "goal.json").write_text(json.dumps(goal))
            np.savez(bdir / "init.npz", qpos=st["qpos"], qvel=st["qvel"])
            (bdir / "meta.json").write_text(
                json.dumps({
                    "dim": dim,
                    "suite": suite,
                    "bddl": rel,
                    "nq": st["nq"],
                    "nv": st["nv"],
                    "lang": st["lang"],
                    "goal": goal,
                })
            )
            ok += 1
            rec = {"name": name, "dim": dim, "nq": st["nq"], "goal": [g[0] for g in goal], "ok": True}
            print(f"  [{i + 1:3d}/{len(tasks)}] {name[:64]:64s} nq={st['nq']} goal={[g[0] for g in goal]}", flush=True)
        except Exception as e:
            err += 1
            rec = {"name": name, "dim": dim, "ok": False, "error": str(e)[:160]}
            print(f"  [{i + 1:3d}/{len(tasks)}] {name[:64]:64s} ERROR {str(e)[:70]}", flush=True)
            traceback.print_exc()
        manifest.append(rec)

    assets.close()
    (args.out / "manifest.json").write_text(
        json.dumps(
            {"n": len(tasks), "ok": ok, "errors": err, "fetched_files": assets.fetched, "tasks": manifest}, indent=2
        )
    )
    print(f"\nGEN LIBERO-plus: {ok}/{len(tasks)} bundles | {err} errors | fetched {assets.fetched} asset files")
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
