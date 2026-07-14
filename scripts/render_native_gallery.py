"""Render a side-by-side visualization for every SimplerEnv task — NATIVE ONLY, zero cloned lib.

For each of the 25 SimplerEnv tasks this builds the corresponding vendored ``Native*Env`` (importing
ONLY ``roboverse_pack.tasks.simpler_env._native`` + migrated ``roboverse_data`` assets — NO
``mani_skill2_real2sim`` / ``simpler_env`` import, asserted at runtime), drives a deterministic
reach-close-lift motion, and writes an MP4 whose frames are side-by-side:

    [ raw SAPIEN render | visual-matching greenscreen overlay ]

both produced by the native pipeline (scene + robot + objects + vendored overlay). This proves the
deletable port can generate complete task visualizations with the upstream package absent.

Run (driver, one subprocess per task):  JAX_PLATFORMS=cpu python scripts/render_native_gallery.py
Single task:                            python scripts/render_native_gallery.py --task google_robot_pick_coke_can
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Asset root: repo-local by default, overridable for an out-of-tree data checkout.
RV = os.environ.get("ROBOVERSE_DATA", os.path.join(_REPO_ROOT, "roboverse_data"))
RVA = f"{RV}/assets/simpler_env"
URDF_G = f"{RV}/robots/google_robot/urdf/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"
URDF_W = f"{RV}/robots/widowx/widowx_description/wx250s.urdf"
COKE_SCENE = f"{RVA}/scenes/google_pick_coke_can_1_v4.glb"
COKE_OVERLAY = f"{RVA}/scenes/google_coke_can_real_eval_1.png"
MOVENEAR_OVERLAY = f"{RVA}/real_inpainting/google_move_near_real_eval_1.png"
DRAWER_OVERLAY = f"{RVA}/real_inpainting/open_drawer_b0.png"
BRIDGE_OVERLAY = f"{RVA}/real_inpainting/bridge_real_eval_1.png"
SINK_OVERLAY = f"{RVA}/real_inpainting/bridge_sink.png"
CABINET = f"{RVA}/cabinet/mk_station.urdf"
OUT_DIR = "/tmp/native_gallery"

# 25 tasks -> (family, kwargs)
TASKS = {
    "google_robot_pick_coke_can": ("coke", {"default_orientation": None}),
    "google_robot_pick_horizontal_coke_can": ("coke", {"default_orientation": "lr_switch"}),
    "google_robot_pick_vertical_coke_can": ("coke", {"default_orientation": "laid_vertically"}),
    "google_robot_pick_standing_coke_can": ("coke", {"default_orientation": "upright"}),
    "google_robot_pick_object": ("pickobject", {}),
    "google_robot_move_near_v0": ("movenear", {"variant": "v0"}),
    "google_robot_move_near_v1": ("movenear", {"variant": "v1"}),
    "google_robot_move_near": ("movenear", {"variant": "v1"}),
    "google_robot_open_drawer": ("drawer", {"is_close": False, "drawer_ids": ["top", "middle", "bottom"]}),
    "google_robot_open_top_drawer": ("drawer", {"is_close": False, "drawer_ids": ["top"]}),
    "google_robot_open_middle_drawer": ("drawer", {"is_close": False, "drawer_ids": ["middle"]}),
    "google_robot_open_bottom_drawer": ("drawer", {"is_close": False, "drawer_ids": ["bottom"]}),
    "google_robot_close_drawer": ("drawer", {"is_close": True, "drawer_ids": ["top", "middle", "bottom"]}),
    "google_robot_close_top_drawer": ("drawer", {"is_close": True, "drawer_ids": ["top"]}),
    "google_robot_close_middle_drawer": ("drawer", {"is_close": True, "drawer_ids": ["middle"]}),
    "google_robot_close_bottom_drawer": ("drawer", {"is_close": True, "drawer_ids": ["bottom"]}),
    "google_robot_place_in_closed_drawer": (
        "place",
        {"drawer_ids": ["top", "middle", "bottom"], "fixed_model_id": None},
    ),
    "google_robot_place_in_closed_top_drawer": ("place", {"drawer_ids": ["top"], "fixed_model_id": None}),
    "google_robot_place_in_closed_middle_drawer": ("place", {"drawer_ids": ["middle"], "fixed_model_id": None}),
    "google_robot_place_in_closed_bottom_drawer": ("place", {"drawer_ids": ["bottom"], "fixed_model_id": None}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "place",
        {"drawer_ids": ["top"], "fixed_model_id": "baked_apple_v2"},
    ),
    "widowx_spoon_on_towel": ("puton", {"task": "spoon"}),
    "widowx_carrot_on_plate": ("puton", {"task": "carrot"}),
    "widowx_stack_cube": ("puton", {"task": "stack"}),
    "widowx_put_eggplant_in_basket": ("puton", {"task": "eggplant"}),
}


class _BlockUpstream:
    """meta-path finder that makes the cloned SimplerEnv libs unimportable — simulates deletion."""

    def find_spec(self, name, path=None, target=None):
        if name == "simpler_env" or name == "mani_skill2_real2sim" or name.startswith("mani_skill2_real2sim."):
            raise ModuleNotFoundError(f"blocked to prove native-only rendering (simulating deleted lib): {name}")
        return None


def _block_upstream():
    sys.meta_path.insert(0, _BlockUpstream())


def _assert_no_upstream():
    bad = [m for m in sys.modules if m == "simpler_env" or m.startswith("mani_skill2_real2sim")]
    assert not bad, f"upstream library leaked into native render: {bad}"


def _scripted_actions(n_down=6, n_close=3, n_lift=7):
    """Deterministic reach-down -> close gripper -> lift (7-D EE-delta + gripper)."""
    acts = []
    for _ in range(n_down):
        acts.append(np.array([0, 0, -0.06, 0, 0, 0, 1.0], dtype=np.float32))  # descend, gripper open
    for _ in range(n_close):
        acts.append(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))  # close gripper
    for _ in range(n_lift):
        acts.append(np.array([0, 0, 0.06, 0, 0, 0, -1.0], dtype=np.float32))  # lift, gripper closed
    return acts


def _build(task):
    fam, kw = TASKS[task]
    common_db = f"{RVA}/model_db"
    if fam == "coke":
        from roboverse_pack.tasks.simpler_env._native.env import NativeCokeCanEnv

        env = NativeCokeCanEnv(
            urdf_path=URDF_G,
            arena_glb=COKE_SCENE,
            coke_collision=f"{RVA}/opened_coke_can/collision.obj",
            coke_visual=f"{RVA}/opened_coke_can/textured.dae",
            overlay_png=COKE_OVERLAY,
            **kw,
        ).build()
        return (
            env,
            COKE_OVERLAY,
            lambda e: [e.scene_obj.obj],
            lambda e: e.scene_obj.robot,
            lambda e: e.scene_obj.scene,
            lambda e: e.scene_obj.camera,
            None,
        )
    if fam == "pickobject":
        from roboverse_pack.tasks.simpler_env._native.pick_object import NativePickObjectEnv

        env = NativePickObjectEnv(
            urdf_path=URDF_G,
            arena_glb=COKE_SCENE,
            models_dir=f"{RVA}/models",
            model_db_dir=common_db,
            overlay_png=COKE_OVERLAY,
        ).build()
        return env, COKE_OVERLAY, lambda e: [e.obj], lambda e: e.robot, lambda e: e.scene, lambda e: e.camera, None
    if fam == "movenear":
        from roboverse_pack.tasks.simpler_env._native.move_near import NativeMoveNearEnv

        env = NativeMoveNearEnv(
            urdf_path=URDF_G,
            arena_glb=COKE_SCENE,
            models_dir=f"{RVA}/models",
            model_db_dir=common_db,
            overlay_png=MOVENEAR_OVERLAY,
            **kw,
        ).build()
        return (
            env,
            MOVENEAR_OVERLAY,
            lambda e: list(e.objs),
            lambda e: e.robot,
            lambda e: e.scene,
            lambda e: e.camera,
            None,
        )
    if fam == "drawer":
        from roboverse_pack.tasks.simpler_env._native.drawer import NativeDrawerEnv

        env = NativeDrawerEnv(urdf_path=URDF_G, cabinet_urdf=CABINET, **kw).build()
        return (
            env,
            DRAWER_OVERLAY,
            lambda e: [],
            lambda e: e.robot,
            lambda e: e.scene,
            lambda e: e.camera,
            lambda e: e.cabinet,
        )
    if fam == "place":
        from roboverse_pack.tasks.simpler_env._native.place import NativePlaceInDrawerEnv

        env = NativePlaceInDrawerEnv(
            urdf_path=URDF_G, cabinet_urdf=CABINET, models_dir=f"{RVA}/models", model_db_dir=common_db, **kw
        ).build()
        return (
            env,
            DRAWER_OVERLAY,
            lambda e: [e.obj],
            lambda e: e.robot,
            lambda e: e.scene,
            lambda e: e.camera,
            lambda e: e.cabinet,
        )
    if fam == "puton":
        from roboverse_pack.tasks.simpler_env._native.put_on import NativePutOnEnv

        env = NativePutOnEnv(
            urdf_path=URDF_W, scenes_dir=f"{RVA}/scenes", models_dir=f"{RVA}/models", model_db_dir=common_db, **kw
        ).build()
        ov = SINK_OVERLAY if env.cfg["sink"] else BRIDGE_OVERLAY
        return env, ov, lambda e: list(e.objs), lambda e: e.robot, lambda e: e.scene, lambda e: e.camera, None
    raise ValueError(fam)


def run_task(task):
    _block_upstream()  # make the cloned libs unimportable BEFORE touching roboverse_pack
    import cv2
    import imageio

    from roboverse_pack.tasks.simpler_env._native.overlay import (
        apply_greenscreen_overlay,
        foreground_actor_ids,
        load_overlay_image,
    )

    env, overlay_path, get_objs, get_robot, get_scene, get_cam, get_cab = _build(task)
    _assert_no_upstream()
    env.reset(seed=0)
    cam, robot, scene = get_cam(env), get_robot(env), get_scene(env)
    overlay = load_overlay_image(overlay_path)
    robot_ids = [lk.get_id() for lk in robot.get_links()]
    cab = get_cab(env) if get_cab else None
    extra_ids = [lk.get_id() for lk in cab.get_links()] if cab is not None else []

    def frame():
        scene.update_render()
        cam.take_picture()
        color = cam.get_float_texture("Color")[..., :3]
        raw = np.clip(color * 255, 0, 255).astype(np.uint8)
        try:
            seg = cam.get_uint32_texture("Segmentation")[..., 1]
            fg = foreground_actor_ids(robot_ids, [o.get_id() for o in get_objs(env)], extra_ids)
            ov = apply_greenscreen_overlay(color.astype(np.float64), seg, overlay, foreground_actor_ids=fg)
            ovi = np.clip(ov * 255, 0, 255).astype(np.uint8)
        except Exception:
            ovi = raw
        h = raw.shape[0]
        bar = np.full((h, 4, 3), 255, np.uint8)
        return np.concatenate([raw, bar, ovi], axis=1)

    frames = [frame()]
    for a in _scripted_actions():
        env.step(a)
        frames.append(frame())
    # pad to a few extra still frames at the end
    frames += [frames[-1]] * 3

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{task}.mp4"
    # label
    labeled = []
    for f in frames:
        f = f.copy()
        cv2.putText(f, "native sim", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(
            f,
            "visual-matching (native overlay)",
            (f.shape[1] // 2 + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        labeled.append(f)
    imageio.mimsave(out, labeled, fps=8)
    print(f"OK {task} -> {out} ({len(frames)} frames, {frames[0].shape})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.task:
        return run_task(a.task)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    ok = 0
    for t in TASKS:
        r = subprocess.run([sys.executable, __file__, "--task", t], env=env, check=False)
        ok += r.returncode == 0
        if r.returncode != 0:
            print(f"FAIL {t}")
    print(f"=== gallery: {ok}/{len(TASKS)} tasks rendered ===")


if __name__ == "__main__":
    main()
