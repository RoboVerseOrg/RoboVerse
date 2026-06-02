"""Validate ALL 130 LIBERO tasks (5 suites) — engine + state bitwise.

Downloading every dataset is ~100 GB, so we remote-read ONLY each task's tiny
``model_file`` attr + one demo's ``states`` from HuggingFace (HTTP range requests,
no full download), then run the same parity as ``libero_replay``:
  * engine parity  — raw mujoco (LIBERO engine) vs MetaSim dm_control loader,
    controller-free, qpos/qvel bitwise.
  * state replay   — recorded states -> handler FK body poses vs raw mujoco, bitwise.

Usage:
    python -m tools.libero_integration.sweep_all --out <report>/metasim_1to1/all130.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # need network for remote reads
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import fsspec  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .libero_replay import _roll, remap_libero_model  # noqa: E402

_HF = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
_LOCAL = {  # already downloaded in full — read locally
    ("libero_object", "pick_up_the_butter_and_place_it_in_the_basket"): "datasets/libero/butter_demo.hdf5",
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"): "datasets/libero/goal_drawer_demo.hdf5",
}


def _read_demo(suite: str, stem: str):
    local = _LOCAL.get((suite, stem))
    if local and os.path.exists(local):
        f = h5py.File(local, "r")
        opener = None
    else:
        opener = fsspec.open(f"{_HF}/{suite}/{stem}_demo.hdf5", "rb")
        f = h5py.File(opener.open(), "r")
    g = f["data"]["demo_0"]
    mf = g.attrs["model_file"]
    states = g["states"][: min(80, g["states"].shape[0])]
    last_done = int(g["dones"][()][-1]) if "dones" in g else int(g["rewards"][()][-1] >= 1.0)
    f.close()
    return mf, states, last_done


def analyze(suite: str, stem: str, n_engine: int = 40) -> dict:
    mf, states, last_done = _read_demo(suite, stem)
    xml = remap_libero_model(mf)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    nq, nv = m.nq, m.nv
    dec = int(round((1 / 20) / m.opt.timestep))
    q0, v0 = states[0][1 : 1 + nq], states[0][1 + nq : 1 + nq + nv]
    raw = _roll(m, d, q0, v0, dec, n_engine)
    from dm_control import mjcf

    phys = mjcf.Physics.from_mjcf_model(mjcf.from_xml_string(xml, assets=None))
    hm, hd = phys.model.ptr, phys.data.ptr
    ms = _roll(hm, hd, q0, v0, dec, n_engine, override=(m.body_pos.copy(), m.body_quat.copy()))
    eng = float(np.abs(raw - ms).max())
    fk = 0.0
    for s in states:
        qp = s[1 : 1 + nq]
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        hd.qpos[:] = qp
        mujoco.mj_forward(hm, hd)
        fk = max(fk, float(np.abs(d.xpos - hd.xpos).max()))
    return {
        "suite": suite, "task": stem, "nq": int(nq), "nv": int(nv), "nu": int(m.nu),
        "engine_max_abs_diff": eng, "engine_bitwise": bool(eng == 0.0),
        "state_fk_max_abs_diff": fk, "state_bitwise": bool(fk == 0.0), "demo_success": bool(last_done == 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks-json", default="datasets/libero/all_tasks.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tasks = json.loads(Path(args.tasks_json).read_text())
    if args.limit:
        tasks = tasks[: args.limit]

    results, eng_ok, st_ok, succ, errs = [], 0, 0, 0, 0
    by_suite: dict[str, list[bool]] = {}
    for i, (suite, _pf, stem) in enumerate(tasks):
        try:
            r = analyze(suite, stem)
            eng_ok += r["engine_bitwise"]
            st_ok += r["state_bitwise"]
            succ += r["demo_success"]
            by_suite.setdefault(suite, []).append(r["engine_bitwise"] and r["state_bitwise"])
            print(f"  [{i + 1:3d}/{len(tasks)}] {suite:14s} {stem[:42]:42s} eng={r['engine_max_abs_diff']:.0e} "
                  f"st={r['state_fk_max_abs_diff']:.0e} bit={r['engine_bitwise'] and r['state_bitwise']}", flush=True)
        except Exception as e:  # noqa: BLE001
            errs += 1
            r = {"suite": suite, "task": stem, "error": str(e)[:140]}
            print(f"  [{i + 1:3d}/{len(tasks)}] {suite:14s} {stem[:42]:42s} ERROR {str(e)[:60]}", flush=True)
        results.append(r)
        time.sleep(0.3)  # be gentle with HF

    n = len(tasks)
    summary = {
        "n_tasks": n, "engine_bitwise": f"{eng_ok}/{n}", "state_bitwise": f"{st_ok}/{n}",
        "demo_success": f"{succ}/{n}", "errors": errs,
        "per_suite_bitwise": {s: f"{sum(v)}/{len(v)}" for s, v in by_suite.items()},
        "tasks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nALL LIBERO: engine {summary['engine_bitwise']} | state {summary['state_bitwise']} | "
          f"success {summary['demo_success']} | errors {errs}")
    print("per-suite:", summary["per_suite_bitwise"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
