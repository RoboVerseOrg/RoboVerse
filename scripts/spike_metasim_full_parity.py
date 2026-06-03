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
import subprocess
import sys

import numpy as np

RV = "/home/ghr/projects/RoboVerse/RoboVerse-simpler/roboverse_data"
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
    np.savez(f"/tmp/mfp_{which}_{name}.npz", rgbs=np.stack(rgbs), succ=np.array(succ))


def run_compare():
    res = {}
    for name in ALL:
        a = np.load(f"/tmp/mfp_native_{name}.npz")
        b = np.load(f"/tmp/mfp_metasim_{name}.npz")
        ar, br = a["rgbs"].astype(np.float64), b["rgbs"].astype(np.float64)
        k = min(len(ar), len(br))
        res[name] = {
            "rgb": round(float(np.abs(ar[:k] - br[:k]).mean()), 4),
            "succ": bool(np.array_equal(a["succ"], b["succ"])),
        }
    worst = max(r["rgb"] for r in res.values())
    ok = all(r["rgb"] < 1.0 and r["succ"] for r in res.values())
    out = {"all_ok": ok, "n": len(res), "worst_rgb_mean_abs": worst, "per_task": res}
    print("MFP_B64:" + base64.b64encode(json.dumps(out).encode()).decode())
    with open("/tmp/metasim_full_parity.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "metasim", "compare", "all"], default="all")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.mode in ("native", "metasim"):
        return run(a.mode, a.task)
    if a.mode == "compare":
        return run_compare()
    env = dict(os.environ, JAX_PLATFORMS="cpu", LOGURU_LEVEL="WARNING")
    for name in ALL:
        for m in ("native", "metasim"):
            if (
                subprocess.run([sys.executable, __file__, "--mode", m, "--task", name], env=env, check=False).returncode
                != 0
            ):
                print(f"FAIL {name} {m}")
                sys.exit(1)
    subprocess.run([sys.executable, __file__, "--mode", "compare"], env=env, check=False)


if __name__ == "__main__":
    main()
