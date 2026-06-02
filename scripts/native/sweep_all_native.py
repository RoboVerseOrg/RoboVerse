"""All-130 LIBERO native 1:1 sweep + side-by-side visualization.

For every base LIBERO task (5 suites, 130 tasks) this:

1. Builds the native LIBERO env straight from its BDDL file (no benchmark
   registry) and replays the demo's recorded **actions** to capture the per-step
   gripper ctrl + the action-replay success (honest open-loop number).
2. **State-replays** the demo's recorded states and checks the *ported* BDDL
   checker (no libero) against ``env._check_success`` at every step — the bitwise
   1:1 success criterion — while rendering the ground-truth agentview frames.
3. Exports a self-contained bundle (``model.mjb`` + states + actions + grip +
   resolved goal/OSC cfg) to a temp dir, then loads it through
   :class:`LiberoNativeTask` (imports only mujoco+numpy) and:
   * state-replays it to render the native agentview frames + re-check the ported
     goal (proves render + checker run with no libero), measuring native↔GT pixel
     MAE at identical states;
   * runs the inlined OSC_POSE controller open-loop to reach success natively.
4. Composes a right-side-up side-by-side video (libero GT | MetaSim-native) and a
   mid-trajectory thumbnail, then deletes the temp ``model.mjb`` (234 MB each).

Run (liberoplus env):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero_plus PYTHONPATH=. \\
    python -m scripts.native.sweep_all_native --out outputs/native_sweep [--limit N] [--suites libero_object,...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from roboverse_pack.tasks.libero_native import checker as nc
from roboverse_pack.tasks.libero_native.native_task import LiberoNativeTask
from scripts.native.export_libero_task import _resolve_goal, build_osc_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BDDL_ROOT = os.path.join(ROOT, "third_party", "LIBERO", "libero", "libero", "bddl_files")
DEMO_ROOT = os.path.join(ROOT, "third_party", "libero_datasets")
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10", "libero_90"]


def _first_true(flags):
    return next((i for i, s in enumerate(flags) if s), None)


def _sample_idx(n, k):
    """Up to ``k`` evenly spaced indices in ``range(n)`` (always include last)."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _panel(img, label, ok, hw):
    """Resize to (hw,hw), draw a label bar + success dot. ``img`` right-side-up."""
    im = Image.fromarray(img).resize((hw, hw), Image.BILINEAR).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, hw, 16], fill=(0, 0, 0))
    d.text((4, 3), label, fill=(255, 255, 255))
    d.ellipse([hw - 14, 2, hw - 4, 12], fill=(60, 200, 90) if ok else (210, 70, 70))
    return np.asarray(im)


def _compose(gt, nat, gt_ok, nat_ok, hw=192):
    left = _panel(gt[::-1], "LIBERO (robosuite)", gt_ok, hw)  # GT obs is bottom-up
    right = _panel(nat, "MetaSim-native (no libero)", nat_ok, hw)
    sep = np.full((hw, 2, 3), 40, np.uint8)  # 2px keeps total width even for h264
    return np.concatenate([left, sep, right], axis=1)


def sweep_task(suite, base, video_dir, thumb_dir, render_hw, video_hw, n_video):
    bddl = os.path.join(BDDL_ROOT, suite, base + ".bddl")
    demo = os.path.join(DEMO_ROOT, suite, base + "_demo.hdf5")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=render_hw, camera_widths=render_hw)
    env.reset()
    e = env.env
    goal = _resolve_goal(e)
    cfg = build_osc_cfg(e)
    pred_types = sorted({g[0].lower() for g in e.parsed_problem["goal_state"]})
    m, d = e.sim.model._model, e.sim.data._data

    with h5py.File(demo, "r") as h:
        states = np.asarray(h["data"]["demo_0"]["states"])
        actions = np.asarray(h["data"]["demo_0"]["actions"])
    nstep = len(actions)
    vidx = _sample_idx(len(states), n_video)

    # --- pass 1: action replay (capture grip ctrl + action-replay GT success) ---
    env.set_init_state(states[0])
    grip_ctrl, act_succ = [], []
    for a in actions:
        env.step(a.tolist())
        grip_ctrl.append(np.array(e.sim.data.ctrl[cfg["grip_act_index"]]))
        act_succ.append(bool(e._check_success()))
    grip_ctrl = np.array(grip_ctrl)

    # --- pass 2: state replay -> ported-vs-native checker parity + GT frames ---
    mism, gt_state_succ = 0, []
    gt_frames = {}
    for i, st in enumerate(states):
        obs = env.regenerate_obs_from_state(st)
        sn = bool(e._check_success())
        sp = bool(nc.check_success(m, d, goal))
        mism += int(sn != sp)
        gt_state_succ.append(sn)
        if i in vidx:
            gt_frames[i] = obs["agentview_image"].copy()

    # --- export self-contained bundle to a temp dir ---
    tmp = tempfile.mkdtemp(prefix="native_")
    mujoco.mj_saveModel(m, os.path.join(tmp, "model.mjb"), None)
    np.save(os.path.join(tmp, "init_states.npy"), states[0])
    np.save(os.path.join(tmp, "demo_actions.npy"), actions)
    np.save(os.path.join(tmp, "grip_ctrl.npy"), grip_ctrl)
    json.dump({"goal": goal, "suite": suite, "base": base, "osc": cfg}, open(os.path.join(tmp, "goal.json"), "w"))
    env.close()  # free the robosuite EGL context before building the native renderer

    # --- native: load from MJB (no libero), state-replay render+checker, OSC rollout ---
    task = LiberoNativeTask(base, bundle_dir=tmp)
    nat_mism, nat_state_succ, maes = 0, [], []
    nat_frames = {}
    for i, st in enumerate(states):
        task.reset(st)
        sp = bool(task.success())
        nat_state_succ.append(sp)
        sg = bool(gt_state_succ[i])
        nat_mism += int(sp != sg)
        if i in vidx:
            f = task.render(height=render_hw, width=render_hw)
            nat_frames[i] = f
            maes.append(float(np.abs(f.astype(float) - gt_frames[i][::-1].astype(float)).mean()))
    # OSC open-loop rollout (native drives itself) -> first success step
    osc = task.replay_demo()
    if getattr(task, "_renderer", None) is not None:
        task._renderer.close()

    # --- compose side-by-side video + thumbnail ---
    frames = [_compose(gt_frames[i], nat_frames[i], gt_state_succ[i], nat_state_succ[i], video_hw) for i in vidx]
    tag = f"{suite}__{base}"
    vid_path = os.path.join(video_dir, tag + ".mp4")
    imageio.mimsave(vid_path, frames, fps=12, quality=6, macro_block_size=1)
    mid = vidx[len(vidx) * 2 // 3]
    imageio.imwrite(
        os.path.join(thumb_dir, tag + ".png"),
        _compose(gt_frames[mid], nat_frames[mid], gt_state_succ[mid], nat_state_succ[mid], video_hw),
    )
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "suite": suite,
        "task": base,
        "predicates": pred_types,
        "n_goal_preds": len(goal),
        "n_demo_steps": nstep,
        "state_replay_checker_mismatches": mism,  # ported vs env._check_success (1:1)
        "native_vs_gt_checker_mismatches": nat_mism,  # native(MJB) vs GT state-replay
        "gt_state_replay_success": bool(any(gt_state_succ)),
        "native_state_replay_success": bool(any(nat_state_succ)),
        "gt_action_replay_success": bool(any(act_succ)),
        "native_osc_first_success_step": osc["first_success_step"],
        "native_osc_final_success": bool(osc["final_success"]),
        "render_mae_mean": round(float(np.mean(maes)), 3) if maes else None,
        "render_mae_max": round(float(np.max(maes)), 3) if maes else None,
        "video": os.path.relpath(vid_path, ROOT),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/native_sweep")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--limit", type=int, default=0, help="max tasks per suite (0=all)")
    ap.add_argument("--render_hw", type=int, default=256)
    ap.add_argument("--video_hw", type=int, default=192)
    ap.add_argument("--n_video", type=int, default=48)
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out)
    vdir, tdir = os.path.join(out, "videos"), os.path.join(out, "thumbs")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    results_path = os.path.join(out, "results.json")
    prior = json.load(open(results_path)) if os.path.isfile(results_path) else []
    results = [r for r in prior if "error" not in r]  # drop prior errors so they re-run
    done = {(r["suite"], r["task"]) for r in results}

    for suite in args.suites.split(","):
        bases = sorted(
            f[: -len("_demo.hdf5")] for f in os.listdir(os.path.join(DEMO_ROOT, suite)) if f.endswith("_demo.hdf5")
        )
        if args.limit:
            bases = bases[: args.limit]
        for i, base in enumerate(bases):
            if (suite, base) in done:
                print(f"[skip] {suite}/{base}")
                continue
            try:
                r = sweep_task(suite, base, vdir, tdir, args.render_hw, args.video_hw, args.n_video)
                results.append(r)
                json.dump(results, open(results_path, "w"), indent=2)
                print(
                    f"[ok] {suite:14s} {i + 1:3d} {base[:46]:46s} preds={r['predicates']} "
                    f"chk_mism={r['state_replay_checker_mismatches']} nat_mism={r['native_vs_gt_checker_mismatches']} "
                    f"mae={r['render_mae_mean']} osc_succ={r['native_osc_first_success_step']}"
                )
            except Exception as ex:
                import traceback

                results.append({"suite": suite, "task": base, "error": f"{type(ex).__name__}: {ex}"})
                json.dump(results, open(results_path, "w"), indent=2)
                print(f"[FAIL] {suite}/{base}: {type(ex).__name__}: {ex}")
                traceback.print_exc()

    ok = [r for r in results if "error" not in r]
    print(f"\n=== sweep done: {len(ok)}/{len(results)} tasks ok ===")
    print(f"checker 1:1 (0 mismatch): {sum(1 for r in ok if r['state_replay_checker_mismatches'] == 0)}/{len(ok)}")
    print(f"native==GT checker      : {sum(1 for r in ok if r['native_vs_gt_checker_mismatches'] == 0)}/{len(ok)}")
    print(f"GT state-replay success : {sum(1 for r in ok if r['gt_state_replay_success'])}/{len(ok)}")


if __name__ == "__main__":
    raise SystemExit(main())
