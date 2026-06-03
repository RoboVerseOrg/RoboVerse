"""Vendor + verify ALL 130 LIBERO tasks as native MetaSim bundles.

For every task: remote-read its model_file + a demo's initial/final states +
recorded done from HF (range requests, no full download); read the BDDL goal
locally; write a native bundle (model.xml + goal.json + init.npz); load the model
and check that the native BDDL success at the demo's final state equals the
recorded done. One pass vendors all bundles AND validates the native predicates.

    python -m tools.libero_integration.vendor_all_native --out <report>/metasim_1to1/native130.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import fsspec  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from roboverse_pack.tasks.libero._native_util import check_bddl_success, parse_goal, remap_libero_model  # noqa: E402

_HF = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
_BDDL = "/venv/roboverse/lib/python3.11/site-packages/libero/libero/bddl_files"
_BUNDLES = Path(__file__).resolve().parents[2] / "roboverse_pack/tasks/libero/native_bundles"
_ASSETS = str(Path(__file__).resolve().parents[2] / "third_party/LIBERO/libero/libero/assets")
_LOCAL = {
    ("libero_object", "pick_up_the_butter_and_place_it_in_the_basket"): "datasets/libero/butter_demo.hdf5",
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"): "datasets/libero/goal_drawer_demo.hdf5",
}


def _read(suite, stem):
    local = _LOCAL.get((suite, stem))
    f = h5py.File(local, "r") if local and os.path.exists(local) else h5py.File(fsspec.open(f"{_HF}/{suite}/{stem}_demo.hdf5", "rb").open(), "r")
    g = f["data"]["demo_0"]
    mf = g.attrs["model_file"]
    s0 = g["states"][0]
    sN = g["states"][-1]
    done = int(g["dones"][()][-1]) if "dones" in g else int(g["rewards"][()][-1] >= 1.0)
    f.close()
    return mf, s0, sN, done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks-json", default="datasets/libero/all_tasks.json")
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks_json).read_text())

    results, vendored, matched, errs = [], 0, 0, 0
    by_suite_match: dict[str, list[bool]] = {}
    for i, (suite, pf, stem) in enumerate(tasks):
        name = f"{suite.replace('libero_', '')}__{stem}"[:120]
        try:
            mf, s0, sN, done = _read(suite, stem)
            xml = remap_libero_model(mf, _ASSETS)
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            nq, nv = m.nq, m.nv
            goal = parse_goal(Path(f"{_BDDL}/{pf}/{stem}.bddl").read_text())
            # write bundle
            out = _BUNDLES / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.xml").write_text(mf)
            (out / "goal.json").write_text(json.dumps(goal))
            np.savez(out / "init.npz", qpos=s0[1 : 1 + nq], qvel=s0[1 + nq : 1 + nq + nv])
            vendored += 1
            # verify native success at demo-final == recorded done
            d.qpos[:] = sN[1 : 1 + nq]
            d.qvel[:] = sN[1 + nq : 1 + nq + nv]
            native = check_bddl_success(m, d, goal)
            ok = native == bool(done)
            matched += ok
            by_suite_match.setdefault(suite, []).append(ok)
            rec = {"name": name, "suite": suite, "goal": goal, "native_success": native, "recorded_done": done, "match": ok}
            print(f"  [{i + 1:3d}/130] {name[:60]:60s} goal={[g[0] for g in goal]} native={native} done={done} match={ok}", flush=True)
        except Exception as e:  # noqa: BLE001
            errs += 1
            rec = {"name": name, "suite": suite, "error": str(e)[:140]}
            print(f"  [{i + 1:3d}/130] {name[:60]:60s} ERROR {str(e)[:60]}", flush=True)
        results.append(rec)
        time.sleep(0.25)

    summary = {
        "n": len(tasks), "vendored": vendored, "success_match": f"{matched}/{len(tasks)}", "errors": errs,
        "per_suite_match": {s: f"{sum(v)}/{len(v)}" for s, v in by_suite_match.items()},
        "tasks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nNATIVE 130: vendored {vendored}/{len(tasks)} | success-match {summary['success_match']} | errors {errs}")
    print("per-suite match:", summary["per_suite_match"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
