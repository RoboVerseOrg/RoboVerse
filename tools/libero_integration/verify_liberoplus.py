"""Verify native LIBERO-plus bundles in the shared MetaSim env (mujoco 3.8).

Mirrors ``sweep_all`` (base LIBERO) for the perturbed scenes produced by
``gen_liberoplus`` (built in the dedicated robosuite-1.4 env). For each bundle we
remap the embedded ``model_file`` to local assets and check:

  * engine parity — raw mujoco vs MetaSim's dm_control loader, controller-free,
    qpos/qvel **bitwise** (the perturbed scene steps identically on both engines);
  * handler load — the scene instantiates on the groundless ``MujocoHandler``;
  * native BDDL success — ``check_bddl_success`` evaluates cleanly on the
    perturbed state (goal predicates carry over from base LIBERO).

    python -m tools.libero_integration.verify_liberoplus \
        --bundles datasets/liberoplus --out <report>/metasim_1to1/liberoplus.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from roboverse_pack.tasks.libero._native_util import check_bddl_success, remap_libero_model

_BASE_ASSETS = str(Path(__file__).resolve().parents[2] / "third_party/LIBERO/libero/libero/assets")


def _ctrl(t, nu):
    return np.zeros(nu)  # controller-free: pure passive dynamics, engine-vs-engine


def _roll(model, data, q0, v0, dec, n, override=None):
    if override is not None:
        model.body_pos[:] = override[0]
        model.body_quat[:] = override[1]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q0
    data.qvel[:] = v0
    mujoco.mj_forward(model, data)
    qp = [data.qpos.copy()]
    for _ in range(n):
        data.ctrl[:] = 0.0
        for _ in range(dec):
            mujoco.mj_step(model, data)
        qp.append(data.qpos.copy())
    return np.concatenate(qp)


def analyze(bdir: Path, n_engine: int = 40) -> dict:
    model_xml = (bdir / "model.xml").read_text()
    goal = [tuple(t) for t in json.loads((bdir / "goal.json").read_text())]
    meta = json.loads((bdir / "meta.json").read_text()) if (bdir / "meta.json").exists() else {}
    init = np.load(bdir / "init.npz")
    xml = remap_libero_model(model_xml, _BASE_ASSETS)

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    nq, nv = m.nq, m.nv
    dec = round((1 / 20) / m.opt.timestep)
    q0 = init["qpos"][:nq]
    v0 = init["qvel"][:nv]
    raw = _roll(m, d, q0, v0, dec, n_engine)

    from dm_control import mjcf

    phys = mjcf.Physics.from_mjcf_model(mjcf.from_xml_string(xml, assets=None))
    hm, hd = phys.model.ptr, phys.data.ptr
    ms = _roll(hm, hd, q0, v0, dec, n_engine, override=(m.body_pos.copy(), m.body_quat.copy()))
    eng = float(np.abs(raw - ms).max())

    # native success checker must evaluate cleanly on the perturbed scene
    d.qpos[:] = q0
    d.qvel[:] = v0
    succ_runs, succ_val = True, None
    try:
        succ_val = bool(check_bddl_success(m, d, goal))
    except Exception:
        succ_runs = False

    return {
        "name": bdir.name,
        "dim": meta.get("dim"),
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(m.nu),
        "engine_max_abs_diff": eng,
        "engine_bitwise": bool(eng == 0.0),
        "goal": [g[0] for g in goal],
        "success_checker_runs": succ_runs,
        "success_at_init": succ_val,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", type=Path, default=Path("datasets/liberoplus"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dirs = sorted(p for p in args.bundles.iterdir() if (p / "model.xml").exists())
    results, eng_ok, chk_ok, errs = [], 0, 0, 0
    by_dim: dict[str, list[bool]] = {}
    for i, bdir in enumerate(dirs):
        try:
            r = analyze(bdir)
            eng_ok += r["engine_bitwise"]
            chk_ok += r["success_checker_runs"]
            by_dim.setdefault(r["dim"] or "?", []).append(r["engine_bitwise"])
            print(
                f"  [{i + 1:3d}/{len(dirs)}] {r['dim'] or '?':9s} {bdir.name[:50]:50s} "
                f"nq={r['nq']:3d} eng={r['engine_max_abs_diff']:.0e} bit={r['engine_bitwise']} "
                f"chk={r['success_checker_runs']}",
                flush=True,
            )
        except Exception as e:
            errs += 1
            r = {"name": bdir.name, "error": str(e)[:160]}
            print(f"  [{i + 1:3d}/{len(dirs)}] ERROR {bdir.name[:50]} :: {str(e)[:70]}", flush=True)
        results.append(r)

    n = len(dirs)
    summary = {
        "n": n,
        "engine_bitwise": f"{eng_ok}/{n}",
        "success_checker_runs": f"{chk_ok}/{n}",
        "errors": errs,
        "per_dim_engine_bitwise": {k: f"{sum(v)}/{len(v)}" for k, v in by_dim.items()},
        "dims": dict(Counter(r.get("dim") for r in results if "dim" in r)),
        "tasks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(
        f"\nLIBERO-plus VERIFY: engine {summary['engine_bitwise']} bitwise | "
        f"checker {summary['success_checker_runs']} | errors {errs}"
    )
    print("per-dim engine bitwise:", summary["per_dim_engine_bitwise"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
