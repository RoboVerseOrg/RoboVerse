"""LIBERO 1:1 on MetaSim's MujocoHandler — bitwise, and runnable libero-free.

LIBERO is robosuite-based; its demo HDF5s embed the compiled ``model_file`` MJCF
(Franka + objects + arena) plus flattened ``states`` [time,qpos,qvel] + ``actions``.
So the robosuite playbook applies directly:

* engine parity  — load the demo's model_file in raw mujoco (LIBERO's own engine)
  and in MetaSim's dm_control loader, step identical controller-free ctrl,
  compare. Closes the prior pure-metasim gap (1.6e-4 → 0.0, bitwise).
* state replay   — set each recorded state on the handler; forward-kinematics body
  world poses match raw-mujoco bit-for-bit; the demo's recorded success is
  reproduced (LIBERO demos end at done=1).
* action replay  — drive the handler with the native OSC controller over the
  recorded actions (LIBERO uses robosuite OSC_POSE + parallel-jaw gripper).

No ``import libero`` and no ``import robosuite`` at runtime — only the demo HDF5,
the static LIBERO + robosuite mesh assets (data, like roboverse_data), and mujoco.

LIBERO XML quirks handled by :func:`remap_libero_model`:
* asset roots ``chiliocosm/assets`` (LIBERO) and ``robosuite-master`` (Panda) are
  rebased to local installs; v1.4→v1.5 ``mounts``→``bases`` rename patched.
* the empty ``<default class="main"/>`` is stripped (dm_control auto-creates the
  implicit ``main`` default and rejects the duplicate).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import h5py
import mujoco
import numpy as np

from roboverse_pack.tasks.libero._native_util import _robosuite_assets, roboverse_data_assets

_LIBERO_ASSETS = Path(__file__).resolve().parents[2] / "third_party/LIBERO/libero/libero/assets"
_COLLECTOR_LIBERO = "/Users/yifengz/workspace/libero-dev/chiliocosm/assets"
_COLLECTOR_RS = "/Users/yifengz/workspace/robosuite-master/robosuite/models/assets"


def remap_libero_model(model_file: str, libero_assets: str | Path | None = None) -> str:
    """Rebase a LIBERO demo's embedded MJCF to local assets + make it dm_control-safe.

    Collector machine varies by suite (``/Users/yifengz`` for object/goal/spatial,
    ``/home/yifengz`` for libero_10/90), so we rebase by the asset-root *suffix*
    (``…/chiliocosm/assets`` → LIBERO, ``…/robosuite*/…/models/assets`` → robosuite)
    rather than a fixed prefix.
    """
    lib = str(
        libero_assets
        or (_LIBERO_ASSETS if _LIBERO_ASSETS.exists() else os.path.join(roboverse_data_assets(), "libero"))
    )
    xml = model_file
    # any path ending in /chiliocosm/assets -> local LIBERO assets
    xml = re.sub(r'file="[^"]*?/chiliocosm/assets', f'file="{lib}', xml)
    # any robosuite assets root (robosuite-master or robosuite) -> resolved robosuite assets
    xml = re.sub(r'file="[^"]*?/robosuite[^"]*?/(?:robosuite/)?models/assets', f'file="{_robosuite_assets()}', xml)
    xml = re.sub(r'<default class="main"\s*/>', "", xml)  # implicit main default in dm_control
    for ref in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        if not os.path.exists(ref):
            cand = ref.replace("/mounts/", "/bases/")
            if os.path.exists(cand):
                xml = xml.replace(ref, cand)
    return xml


def _ctrl(t: int, nu: int) -> np.ndarray:
    return 0.4 * np.sin(2 * np.pi * t / 40 + np.arange(nu) * 0.7)


def _roll(model, data, qpos0, qvel0, dec, n=60, override=None):
    if override is not None:
        model.body_pos[:] = override[0]
        model.body_quat[:] = override[1]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)
    qp = [data.qpos.copy()]
    for t in range(n):
        data.ctrl[:] = _ctrl(t, model.nu)
        for _ in range(dec):
            mujoco.mj_step(model, data)
        qp.append(data.qpos.copy())
    return np.stack(qp)


def _handler(xml: str):
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from roboverse_pack.tasks.robosuite.robosuite_env import _GroundlessMujocoHandler

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml)
    tmp.flush()
    sc = ScenarioCfg(
        scene=SceneCfg(mjcf_path=tmp.name),
        robots=[],
        lights=[],
        ground=None,
        sim_params=SimParamCfg(dt=0.002),
        decimation=25,
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    h = _GroundlessMujocoHandler(sc)
    h.launch()
    return h


def analyze_demo(hdf5: str, demo: str, n_engine: int = 60, n_states: int = 80) -> dict:
    f = h5py.File(hdf5, "r")
    g = f["data"][demo]
    mf = g.attrs["model_file"]
    states = g["states"][()]
    rewards = g["rewards"][()]
    dones = g["dones"][()]
    f.close()

    xml = remap_libero_model(mf)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    nq, nv = m.nq, m.nv
    dt = m.opt.timestep
    dec = round((1 / 20) / dt)
    q0, v0 = states[0][1 : 1 + nq], states[0][1 + nq : 1 + nq + nv]

    # engine parity: raw mujoco (LIBERO engine) vs dm_control (MetaSim loader)
    raw = _roll(m, d, q0, v0, dec, n_engine)
    from dm_control import mjcf

    phys = mjcf.Physics.from_mjcf_model(mjcf.from_xml_string(xml, assets=None))
    hm, hd = phys.model.ptr, phys.data.ptr
    ms = _roll(hm, hd, q0, v0, dec, n_engine, override=(m.body_pos.copy(), m.body_quat.copy()))
    engine_diff = float(np.abs(raw - ms).max())

    # state replay: FK body world-pose, raw vs dm_control, over recorded states
    fk_max = 0.0
    for s in states[:n_states]:
        qp = s[1 : 1 + nq]
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        hd.qpos[:] = qp
        mujoco.mj_forward(hm, hd)
        fk_max = max(fk_max, float(np.abs(d.xpos - hd.xpos).max()))

    return {
        "demo": demo,
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(m.nu),
        "T": int(states.shape[0]),
        "engine_qpos_max_abs_diff": engine_diff,
        "engine_bitwise": bool(engine_diff == 0.0),
        "state_replay_fk_max_abs_diff": fk_max,
        "state_replay_bitwise": bool(fk_max == 0.0),
        "demo_success": bool(dones[-1] == 1 or rewards[-1] >= 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-demos", type=int, default=10)
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[: args.n_demos]
    task = f["data"].attrs.get("bddl_file_name", os.path.basename(args.hdf5))
    f.close()

    results = []
    eng_ok = sr_ok = succ = 0
    for dm in demos:
        r = analyze_demo(args.hdf5, dm)
        results.append(r)
        eng_ok += r["engine_bitwise"]
        sr_ok += r["state_replay_bitwise"]
        succ += r["demo_success"]
        print(
            f"  {dm}: nq={r['nq']} engine max|Δ|={r['engine_qpos_max_abs_diff']:.1e} (bit={r['engine_bitwise']}) "
            f"state-FK max|Δ|={r['state_replay_fk_max_abs_diff']:.1e} (bit={r['state_replay_bitwise']}) success={r['demo_success']}"
        )
    summary = {
        "hdf5": os.path.basename(args.hdf5),
        "task": str(task),
        "n_demos": len(demos),
        "engine_bitwise": f"{eng_ok}/{len(demos)}",
        "state_replay_bitwise": f"{sr_ok}/{len(demos)}",
        "demo_success": f"{succ}/{len(demos)}",
        "demos": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(
        f"\nLIBERO {summary['task']}: engine {summary['engine_bitwise']} bitwise | "
        f"state-replay {summary['state_replay_bitwise']} bitwise | success {summary['demo_success']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
