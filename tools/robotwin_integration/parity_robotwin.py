"""Measure RoboTwin <-> RoboVerse replay parity on the achieved joint state.

This is the *measurement* half of the RoboTwin 1:1 claim. It runs in the
``roboverse`` env. Given a bridge pickle collected by ``collect_bridge.py`` (which
now records, per frame, RoboTwin's command-target ``vectors`` AND its *achieved*
``real_vectors`` = ``entity.get_qpos()``), it:

1. converts the trajectory to RoboVerse ``*_v2`` via the shared converter,
2. replays the command-target stream on SAPIEN3 with the ALOHA-AgileX embodiment
   (dof-position-target driven, exactly as ``10_robotwin_aloha_replay.py``),
3. reads RoboVerse's *achieved* joint positions after each step, and
4. reports the per-joint delta between **RoboVerse-achieved and RoboTwin-achieved**
   qpos over the episode (the honest, non-circular parity number), plus the
   RoboVerse-achieved-vs-command-target tracking error for reference.

Both backends are SAPIEN3 with the same URDF, so a tight delta proves the replay
reproduces RoboTwin's joint trajectory. It does NOT prove dynamical equivalence
or task success (this is open-loop state replay) -- see the limitations in the
integration report.

Run (roboverse env)::

    MUJOCO_GL=egl python tools/robotwin_integration/parity_robotwin.py \\
        --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl
    # or sweep every collected bridge pickle:
    MUJOCO_GL=egl python tools/robotwin_integration/parity_robotwin.py --all
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import glob
import json
import os
import pickle

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_handler
from roboverse_pack.robots.aloha_agilex_cfg import AlohaAgilexCfg
from roboverse_pack.tasks.robotwin._convert import CONTROLLED_ARM, ROBOT_NAME, bridge_to_v2

# Arm-joint indices inside RoboTwin's 14-D [L_arm(6), L_grip, R_arm(6), R_grip].
_ARM_VEC_IDX = list(range(0, 6)) + list(range(7, 13))


def _replay_and_measure(bridge: dict, sim: str, settle: int, tmp_v2: str) -> dict:
    """Replay one bridge trajectory in RoboVerse and measure achieved-qpos parity."""
    bridge_to_v2(bridge, tmp_v2)
    robot = AlohaAgilexCfg()
    if not robot.urdf_path:
        raise FileNotFoundError("ALOHA-AgileX URDF not found; extract RoboTwin embodiments.zip")
    init_states, all_actions, _ = get_traj(tmp_v2, robot)
    actions = all_actions[0]

    scenario = ScenarioCfg(
        robots=[robot],
        objects=[],
        cameras=[],
        sim_params=SimParamCfg(dt=0.01),
        decimation=4,
        simulator=sim,
        num_envs=1,
        headless=True,
        add_default_ground=True,
    )
    handler = get_handler(scenario)
    handler.set_states([init_states[0]])

    joint_names = handler.get_joint_names(ROBOT_NAME, sort=True)
    arm_cols = [joint_names.index(j) for j in CONTROLLED_ARM]  # 12 arm joints

    def _achieved_arm():
        st = handler.get_states(mode="tensor")
        qpos = st.robots[ROBOT_NAME].joint_pos[0].detach().cpu().numpy()
        return qpos[arm_cols]

    # real_vectors[t] is RoboTwin's achieved qpos at frame t; vectors[0] is the
    # init state, vectors[1:] are the replayed actions. After replaying action i
    # (= command target vectors[i+1]), RoboVerse-achieved should match real_vectors[i+1].
    real = bridge.get("real_vectors") or []
    has_real = len(real) == len(bridge["vectors"]) and len(real) > 0
    rt_target_arm = np.asarray([bridge["vectors"][k][_ARM_VEC_IDX] for k in range(1, len(bridge["vectors"]))])
    rt_achieved_arm = np.asarray([real[k][_ARM_VEC_IDX] for k in range(1, len(real))]) if has_real else None

    rv_arm = []
    for action in actions:
        handler.set_dof_targets([action])
        for _ in range(max(1, settle)):
            handler.simulate()
        rv_arm.append(_achieved_arm())
    handler.close()
    rv_arm = np.asarray(rv_arm)

    n = min(len(rv_arm), len(rt_target_arm))
    d_target = np.abs(rv_arm[:n] - rt_target_arm[:n])  # RoboVerse-achieved vs command target (tracking)
    out = {
        "task": bridge.get("task"),
        "seed": bridge.get("seed"),
        "frames": int(n),
        "has_achieved_qpos": bool(has_real),
        "tracking_vs_target_rad": {"max": float(d_target.max()), "mean": float(d_target.mean())},
    }
    if has_real:
        d_ach = np.abs(rv_arm[:n] - rt_achieved_arm[:n])  # RoboVerse-achieved vs RoboTwin-achieved (1:1)
        out["parity_vs_achieved_rad"] = {
            "max": float(d_ach.max()),
            "mean": float(d_ach.mean()),
            "per_joint_max": [float(x) for x in d_ach.max(axis=0)],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge", default=os.path.expanduser("~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl"))
    ap.add_argument("--all", action="store_true", help="sweep every *.pkl under the bridge dir")
    ap.add_argument("--bridge-dir", default=os.path.expanduser("~/projects/robotwin/data/_rv_bridge"))
    ap.add_argument("--sim", default="sapien3")
    ap.add_argument("--settle", type=int, default=1, help="simulate() calls applied per recorded target")
    ap.add_argument("--out", default="outputs/robotwin_coverage/parity.json")
    args = ap.parse_args(argv)

    if args.all:
        paths = sorted(
            p for p in glob.glob(os.path.join(args.bridge_dir, "*.pkl")) if not os.path.basename(p).startswith("_")
        )
    else:
        paths = [args.bridge]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp_v2 = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_parity_v2.pkl")
    results = []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            print(f"[skip] missing {p}")
            continue
        with open(p, "rb") as f:
            bridge = pickle.load(f)
        try:
            r = _replay_and_measure(bridge, args.sim, args.settle, tmp_v2)
        except Exception as e:
            print(f"[{i + 1}/{len(paths)}] {bridge.get('task')} parity FAILED: {type(e).__name__}: {e}")
            results.append({"task": bridge.get("task"), "status": "error", "error": f"{type(e).__name__}: {e}"})
            continue
        tag = (
            f"parity(achieved) max={r['parity_vs_achieved_rad']['max']:.4f} mean={r['parity_vs_achieved_rad']['mean']:.4f} rad"
            if r["has_achieved_qpos"]
            else f"tracking max={r['tracking_vs_target_rad']['max']:.4f} rad (no achieved qpos in pkl)"
        )
        print(f"[{i + 1}/{len(paths)}] {r['task']}: {r['frames']} frames | {tag}")
        results.append(r)
        with open(args.out, "w") as f:
            json.dump({"sim": args.sim, "settle": args.settle, "results": results}, f, indent=2)

    ok = [r for r in results if r.get("has_achieved_qpos")]
    if ok:
        worst = max(r["parity_vs_achieved_rad"]["max"] for r in ok)
        mean = float(np.mean([r["parity_vs_achieved_rad"]["mean"] for r in ok]))
        print(f"\n=== PARITY (RoboVerse-achieved vs RoboTwin-achieved), {len(ok)} tasks ===")
        print(f"worst per-frame per-joint max = {worst:.4f} rad ; mean-of-means = {mean:.4f} rad -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
