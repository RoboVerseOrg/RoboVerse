"""Exhaustive MetaSim-native vs proven _native parity over ALL 25 SimplerEnv tasks.

For every task name: build the MetaSim-API task (via the registry) and the standalone _native env
(verified vs upstream), run the same seed + same fixed action sequence in separate subprocesses,
and compare the RAW overhead render (mean-abs over [0,255]) + the per-step success trajectory.
This is the definitive "is every task aligned" check.

Run:  JAX_PLATFORMS=cpu python scripts/spike_metasim_full_parity.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Asset root: repo-local by default, overridable for an out-of-tree data checkout.
RV = os.environ.get("ROBOVERSE_DATA", os.path.join(_REPO_ROOT, "roboverse_data"))
# Per-run rollout artifacts. Wiped at the start of a full run so a stale .npz from an
# earlier run can never be compared as if it were fresh.
ART = os.environ.get("MFP_ARTIFACT_DIR", os.path.join(tempfile.gettempdir(), "metasim_full_parity"))
RVA = f"{RV}/assets/simpler_env"
URDF_G = f"{RV}/robots/google_robot/urdf/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"
URDF_W = f"{RV}/robots/widowx/widowx_description/wx250s.urdf"
CAB = f"{RVA}/cabinet/mk_station.urdf"
COKE_SCENE = f"{RVA}/scenes/google_pick_coke_can_1_v4.glb"
COKE_OV = f"{RVA}/scenes/google_coke_can_real_eval_1.png"
MN_OV = f"{RVA}/real_inpainting/google_move_near_real_eval_1.png"
N = 3
SEED = 1

# task name -> (native-builder key, kwargs for native, metasim handled via registry)
COKE = {
    "google_robot_pick_coke_can": None,
    "google_robot_pick_horizontal_coke_can": "lr_switch",
    "google_robot_pick_vertical_coke_can": "laid_vertically",
    "google_robot_pick_standing_coke_can": "upright",
}
DRAWERS = {
    "google_robot_open_drawer": (False, ["top", "middle", "bottom"]),
    "google_robot_open_top_drawer": (False, ["top"]),
    "google_robot_open_middle_drawer": (False, ["middle"]),
    "google_robot_open_bottom_drawer": (False, ["bottom"]),
    "google_robot_close_drawer": (True, ["top", "middle", "bottom"]),
    "google_robot_close_top_drawer": (True, ["top"]),
    "google_robot_close_middle_drawer": (True, ["middle"]),
    "google_robot_close_bottom_drawer": (True, ["bottom"]),
}
PLACES = {
    "google_robot_place_in_closed_drawer": (["top", "middle", "bottom"], None),
    "google_robot_place_in_closed_top_drawer": (["top"], None),
    "google_robot_place_in_closed_middle_drawer": (["middle"], None),
    "google_robot_place_in_closed_bottom_drawer": (["bottom"], None),
    "google_robot_place_apple_in_closed_top_drawer": (["top"], "baked_apple_v2"),
}
MOVENEAR = {"google_robot_move_near_v0": "v0", "google_robot_move_near_v1": "v1", "google_robot_move_near": "v1"}
PUTON = {
    "widowx_spoon_on_towel": "spoon",
    "widowx_carrot_on_plate": "carrot",
    "widowx_stack_cube": "stack",
    "widowx_put_eggplant_in_basket": "eggplant",
}
ALL = list(COKE) + ["google_robot_pick_object"] + list(MOVENEAR) + list(DRAWERS) + list(PLACES) + list(PUTON)


def _acts(n):
    rng = np.random.RandomState(7)
    return [np.concatenate([rng.uniform(-0.02, 0.02, 6), [(-1.0) ** i]]).astype(np.float32) for i in range(n)]


def _native(name):
    if name in COKE:
        from roboverse_pack.tasks.simpler_env._native.env import NativeCokeCanEnv

        return NativeCokeCanEnv(
            default_orientation=COKE[name],
            urdf_path=URDF_G,
            arena_glb=COKE_SCENE,
            coke_collision=f"{RVA}/opened_coke_can/collision.obj",
            coke_visual=f"{RVA}/opened_coke_can/textured.dae",
            overlay_png=COKE_OV,
        ).build()
    if name == "google_robot_pick_object":
        from roboverse_pack.tasks.simpler_env._native.pick_object import NativePickObjectEnv

        return NativePickObjectEnv(
            urdf_path=URDF_G, arena_glb=COKE_SCENE, models_dir=f"{RVA}/models", model_db_dir=f"{RVA}/model_db"
        ).build()
    if name in MOVENEAR:
        from roboverse_pack.tasks.simpler_env._native.move_near import NativeMoveNearEnv

        return NativeMoveNearEnv(
            variant=MOVENEAR[name],
            urdf_path=URDF_G,
            arena_glb=COKE_SCENE,
            models_dir=f"{RVA}/models",
            model_db_dir=f"{RVA}/model_db",
            overlay_png=MN_OV,
        ).build()
    if name in DRAWERS:
        from roboverse_pack.tasks.simpler_env._native.drawer import NativeDrawerEnv

        ic, dids = DRAWERS[name]
        return NativeDrawerEnv(is_close=ic, drawer_ids=dids, urdf_path=URDF_G, cabinet_urdf=CAB).build()
    if name in PLACES:
        from roboverse_pack.tasks.simpler_env._native.place import NativePlaceInDrawerEnv

        dids, fixed = PLACES[name]
        return NativePlaceInDrawerEnv(
            drawer_ids=dids,
            fixed_model_id=fixed,
            urdf_path=URDF_G,
            cabinet_urdf=CAB,
            models_dir=f"{RVA}/models",
            model_db_dir=f"{RVA}/model_db",
        ).build()
    if name in PUTON:
        from roboverse_pack.tasks.simpler_env._native.put_on import NativePutOnEnv

        return NativePutOnEnv(
            task=PUTON[name],
            urdf_path=URDF_W,
            scenes_dir=f"{RVA}/scenes",
            models_dir=f"{RVA}/models",
            model_db_dir=f"{RVA}/model_db",
        ).build()
    raise ValueError(name)


def _metasim(name):
    from roboverse_pack.tasks.simpler_env._metasim.registry import TASK_MAP

    cls, kw = TASK_MAP[name]
    return cls(**kw)


def run(which, name):
    env = _native(name) if which == "native" else _metasim(name)
    env.reset(seed=SEED)
    rc = env.render_color if hasattr(env, "render_color") else env.scene_obj.render_color

    def raw():
        return np.clip(rc() * 255, 0, 255).astype(np.uint8)

    rgbs, succ = [raw()], []
    for a in _acts(N):
        out = env.step(a)
        rgbs.append(raw())
        succ.append(bool(out[-1]["success"]))
    os.makedirs(ART, exist_ok=True)
    np.savez(os.path.join(ART, f"mfp_{which}_{name}.npz"), rgbs=np.stack(rgbs), succ=np.array(succ))


def run_compare() -> bool:
    """Compare the two sides' recorded rollouts. Returns True only if every task matches.

    Rollouts must be the *same length*: truncating to ``min(len(a), len(b))`` would let a
    side that died after one frame "match" over the frames it did produce. A missing
    artifact raises (FileNotFoundError) rather than silently shrinking the comparison.
    """
    res = {}
    for name in ALL:
        a = np.load(os.path.join(ART, f"mfp_native_{name}.npz"))
        b = np.load(os.path.join(ART, f"mfp_metasim_{name}.npz"))
        ar, br = a["rgbs"].astype(np.float64), b["rgbs"].astype(np.float64)
        same_len = ar.shape == br.shape and len(a["succ"]) == len(b["succ"])
        res[name] = {
            "rgb": round(float(np.abs(ar - br).mean()), 4) if same_len else float("inf"),
            "succ": same_len and bool(np.array_equal(a["succ"], b["succ"])),
            "same_length": bool(same_len),
        }
    worst = max(r["rgb"] for r in res.values())
    # An empty task list proves nothing: `all([])` is True, which would be a vacuous PASS.
    ok = bool(res) and all(r["same_length"] and r["rgb"] < 1.0 and r["succ"] for r in res.values())
    out = {"all_ok": ok, "n": len(res), "worst_rgb_mean_abs": worst, "per_task": res}
    print("MFP_B64:" + base64.b64encode(json.dumps(out).encode()).decode())
    with open(os.path.join(ART, "metasim_full_parity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} — {len(res)} tasks, worst rgb mean-abs {worst}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "metasim", "compare", "all"], default="all")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.mode in ("native", "metasim"):
        run(a.mode, a.task)
        sys.exit(0)
    if a.mode == "compare":
        sys.exit(0 if run_compare() else 1)
    # Full run: start from a clean artifact dir so nothing stale can be compared.
    shutil.rmtree(ART, ignore_errors=True)
    os.makedirs(ART, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS="cpu", LOGURU_LEVEL="WARNING", MFP_ARTIFACT_DIR=ART)
    for name in ALL:
        for m in ("native", "metasim"):
            if (
                subprocess.run([sys.executable, __file__, "--mode", m, "--task", name], env=env, check=False).returncode
                != 0
            ):
                print(f"FAIL {name} {m}")
                sys.exit(1)
    sys.exit(subprocess.run([sys.executable, __file__, "--mode", "compare"], env=env, check=False).returncode)


if __name__ == "__main__":
    main()
