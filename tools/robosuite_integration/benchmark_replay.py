"""Replay robosuite's ORIGINAL benchmark demonstrations 1:1 on MetaSim.

Loads a robomimic/robosuite demonstration HDF5 (the canonical benchmark demos:
per-demo embedded ``model_file`` MJCF + flattened ``states`` [time,qpos,qvel] +
high-level ``actions``) and reproduces them on MetaSim's MujocoHandler:

* STATE replay (exact, the method robosuite itself recommends): set each
  recorded ``qpos/qvel`` and forward. We verify the handler's forward-kinematics
  (body world poses) match a raw-mujoco load of the same MJCF bit-for-bit — so
  the demonstrated motion renders identically. Done for every demo.

* ACTION replay: rebuild the robosuite env from the demo's MJCF, step the
  recorded high-level actions through the OSC controller (capturing per-substep
  torques), then replay those torques on the handler. robosuite vs MetaSim match
  bit-for-bit, i.e. the arm follows the recorded actions identically on both.

Usage:
    python -m tools.robosuite_integration.benchmark_replay \
        --hdf5 datasets/lift_ph_low_dim_v141.hdf5 --out <report>/runs/benchmark_lift \
        --n-demos 200 --render 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def _iter_demos(path: str, n: int | None):
    f = h5py.File(path, "r")
    data = f["data"]
    env_meta = json.loads(data.attrs["env_args"])
    demos = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    if n:
        demos = demos[:n]
    for dm in demos:
        g = data[dm]
        yield dm, g.attrs["model_file"], g["states"][()], g["actions"][()], env_meta
    f.close()


def remap_assets(raw_xml: str, fixer) -> str:
    """Remap a dataset demo's embedded MJCF to the locally-installed robosuite.

    ``fixer.edit_model_xml`` rebases the collector's absolute paths to the local
    install; we then patch the v1.4→v1.5 directory rename (``mounts`` → ``bases``)
    for any file ref that still doesn't resolve.
    """
    import os
    import re

    xml = fixer.edit_model_xml(raw_xml)
    for ref in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        if os.path.exists(ref):
            continue
        cand = ref.replace("/mounts/", "/bases/")
        if os.path.exists(cand):
            xml = xml.replace(ref, cand)
    return xml


def _handler_for(xml: str):
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


def _handler_state_frames(xml: str, qpos_seq: np.ndarray, camera: str = "agentview") -> list:
    """Render a qpos sequence using MetaSim's handler-loaded model (not raw mujoco)."""
    handler = _handler_for(xml)
    hm, hd = handler.physics.model.ptr, handler.physics.data.ptr
    renderer = mujoco.Renderer(hm, height=384, width=384)
    cam = camera if any(
        mujoco.mj_id2name(hm, mujoco.mjtObj.mjOBJ_CAMERA, i) == camera for i in range(hm.ncam)
    ) else 0
    frames = []
    for qpos in qpos_seq:
        hd.qpos[:] = qpos
        mujoco.mj_forward(hm, hd)
        renderer.update_scene(hd, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    handler.close()
    return frames


def state_replay_check(xml: str, states: np.ndarray) -> dict:
    """Set every recorded state on a raw-mujoco model and on the handler; compare FK."""
    raw = mujoco.MjModel.from_xml_string(xml)
    rd = mujoco.MjData(raw)
    nq, nv = raw.nq, raw.nv
    handler = _handler_for(xml)
    hm, hd = handler.physics.model.ptr, handler.physics.data.ptr

    max_xpos = 0.0
    for s in states:
        qpos, qvel = s[1 : 1 + nq], s[1 + nq : 1 + nq + nv]
        rd.qpos[:] = qpos
        rd.qvel[:] = qvel
        mujoco.mj_forward(raw, rd)
        with handler.physics.reset_context():
            hd.qpos[:] = qpos
            hd.qvel[:] = qvel
        max_xpos = max(max_xpos, float(np.abs(rd.xpos - hd.xpos).max()))
    handler.close()
    return {"n_states": int(states.shape[0]), "fk_body_xpos_max_abs_diff": max_xpos}


def action_replay(states: np.ndarray, actions: np.ndarray, env_name: str = "Lift") -> dict:
    """Replay the benchmark's recorded ACTIONS and reproduce them on the handler.

    We rebuild a *fresh* robosuite 1.5.2 env (so its model has the current site
    layout — the dataset's v1.4.1 MJCF would break 1.5.2 env code), transplant the
    demo's initial flattened state (qpos/qvel match: same joint order, nq=16 for
    Lift), then step the recorded high-level actions through the OSC controller,
    capturing per-substep torques. Those torques are replayed on MetaSim's
    handler (loaded from the *1.5.2* compiled MJCF). robosuite and MetaSim match
    bit-for-bit, so the arm follows the recorded benchmark actions identically.
    """
    from .inventory import get
    from .robosuite_rollout import make_robosuite_env

    ref = make_robosuite_env(get(env_name))
    ref.reset()
    rm, rd = ref.sim.model._model, ref.sim.data._data
    nq, nv = rm.nq, rm.nv

    # transplant the demo's initial state into the fresh 1.5.2 env
    ref.sim.set_state_from_flattened(states[0])
    ref.sim.forward()
    xml = ref.model.get_xml()  # 1.5.2 compiled MJCF for the handler
    body_pos, body_quat = rm.body_pos.copy(), rm.body_quat.copy()
    qpos0, qvel0 = rd.qpos.copy(), rd.qvel.copy()

    substep: list[np.ndarray] = []
    orig = ref._pre_action
    ref._pre_action = lambda a, p=False: (orig(a, p), substep.append(rd.ctrl.copy()))[-1]

    ref_qpos = [qpos0.copy()]
    per_step_ctrl = []
    ref_success = []
    for a in actions:
        substep.clear()
        ref.step(a)
        per_step_ctrl.append(np.stack(substep))
        ref_qpos.append(rd.qpos.copy())
        ref_success.append(bool(ref._check_success()))
    ref_qpos = np.stack(ref_qpos)
    ref.close()

    # replay torques on handler
    handler = _handler_for(xml)
    hm, hd = handler.physics.model.ptr, handler.physics.data.ptr
    hm.body_pos[:] = body_pos
    hm.body_quat[:] = body_quat
    with handler.physics.reset_context():
        hd.qpos[:] = qpos0
        hd.qvel[:] = qvel0
    ms_qpos = [hd.qpos.copy()]
    for ctrls in per_step_ctrl:
        for c in ctrls:
            hd.ctrl[:] = c
            mujoco.mj_step(hm, hd)
        ms_qpos.append(hd.qpos.copy())
    ms_qpos = np.stack(ms_qpos)
    handler.close()

    return {
        "n_actions": int(actions.shape[0]),
        "action_replay_qpos_max_abs_diff": float(np.abs(ref_qpos - ms_qpos).max()),
        "action_replay_bitwise": bool((ref_qpos == ms_qpos).all()),
        "action_replay_success": bool(ref_success[-1]) if ref_success else False,
        "_ref_qpos": ref_qpos,
        "_ms_qpos": ms_qpos,
        "_body_pos": body_pos,
        "_body_quat": body_quat,
        "_xml": xml,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env-name", default="Lift")
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--render", type=int, default=0, help="render side-by-side for the first N demos")
    ap.add_argument(
        "--action",
        action="store_true",
        help="also action-replay (needs the dataset's robosuite version to match the installed one)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # one robosuite env to remap each demo's embedded MJCF asset paths
    # (datasets store the collector's absolute paths) to the local install.
    from .inventory import get
    from .robosuite_rollout import make_robosuite_env

    fixer = make_robosuite_env(get(args.env_name))
    fixer.reset()

    per_demo = []
    sr_max = 0.0
    ar_max = 0.0
    ar_bitwise_all = True
    n_success = 0
    rendered = 0
    for i, (dm, raw_xml, states, actions, env_meta) in enumerate(_iter_demos(args.hdf5, args.n_demos)):
        xml = remap_assets(raw_xml, fixer)
        sr = state_replay_check(xml, states)
        sr_max = max(sr_max, sr["fk_body_xpos_max_abs_diff"])
        rec = {"demo": dm, "n_states": sr["n_states"], "fk_body_xpos_max_abs_diff": sr["fk_body_xpos_max_abs_diff"]}
        ar = None
        if args.action:
            ar = action_replay(states, actions, env_name=args.env_name)
            ar_max = max(ar_max, ar["action_replay_qpos_max_abs_diff"])
            ar_bitwise_all = ar_bitwise_all and ar["action_replay_bitwise"]
            n_success += int(ar["action_replay_success"])
            rec.update(
                action_replay_qpos_max_abs_diff=ar["action_replay_qpos_max_abs_diff"],
                action_replay_bitwise=ar["action_replay_bitwise"],
                action_replay_success=ar["action_replay_success"],
            )
        per_demo.append(rec)

        if rendered < args.render:
            from .render import render_qpos_sequence, side_by_side

            raw = mujoco.MjModel.from_xml_string(xml)
            nq = raw.nq
            demo_qpos = states[:, 1 : 1 + nq]
            mdl_state = mujoco.MjModel.from_xml_string(xml)
            # render the demo on raw-mujoco (robosuite's model) and on the handler,
            # both set to the recorded states -> identical real-task motion.
            ms_frames = _handler_state_frames(xml, demo_qpos)
            ref_frames = render_qpos_sequence(mdl_state, demo_qpos, camera="agentview")
            side_by_side(
                ref_frames,
                ms_frames,
                str(args.out / f"{dm}_state_replay.mp4"),
                label_a="robosuite benchmark demo",
                label_b="MetaSim MujocoHandler (1:1)",
            )
            if ar is not None:
                mdl_a = mujoco.MjModel.from_xml_string(ar["_xml"])
                mdl_b = mujoco.MjModel.from_xml_string(ar["_xml"])
                for m in (mdl_a, mdl_b):
                    m.body_pos[:] = ar["_body_pos"]
                    m.body_quat[:] = ar["_body_quat"]
                side_by_side(
                    render_qpos_sequence(mdl_a, ar["_ref_qpos"], camera="agentview"),
                    render_qpos_sequence(mdl_b, ar["_ms_qpos"], camera="agentview"),
                    str(args.out / f"{dm}_action_replay.mp4"),
                    label_a="robosuite follows benchmark actions",
                    label_b="MetaSim handler (1:1)",
                )
            rendered += 1

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1} demos | state-FK max {sr_max:.1e} | action max {ar_max:.1e}")

    summary = {
        "hdf5": os.path.basename(args.hdf5),
        "env_name": args.env_name,
        "n_demos": len(per_demo),
        "total_states": int(sum(d["n_states"] for d in per_demo)),
        "state_replay_fk_max_abs_diff": sr_max,
        "state_replay_bitwise_all": bool(sr_max == 0.0),
        "demos": per_demo,
    }
    if args.action:
        summary.update(
            action_replay_qpos_max_abs_diff=ar_max,
            action_replay_bitwise_all=ar_bitwise_all,
            action_replay_success_count=n_success,
            action_replay_success_rate=n_success / max(len(per_demo), 1),
        )
    (args.out / "benchmark_replay.json").write_text(json.dumps(summary, indent=2))
    msg = (
        f"\n{args.env_name}: {len(per_demo)} benchmark demos ({summary['total_states']} states) | "
        f"state-replay FK max|Δ| {sr_max:.1e} (bitwise={summary['state_replay_bitwise_all']})"
    )
    if args.action:
        msg += f" | action-replay max|Δqpos| {ar_max:.1e} (bitwise={ar_bitwise_all}) | success {n_success}/{len(per_demo)}"
    print(msg)
    print(f"wrote {args.out / 'benchmark_replay.json'}")


if __name__ == "__main__":
    main()
