"""Real-policy success gallery for the 25 MetaSim-native SimplerEnv tasks.

For every task we run a real pretrained policy on the MetaSim-native env
(``SimplerEnv/<task>``) and record the first **successful** episode (the task's
own success checker decides), then compose an mp4 of the visual-matching camera
view — i.e. an actual policy manipulating the objects, not a scripted motion.

Policy per embodiment (matches what SimplerEnv evaluates):
  * Google Robot (21 tasks) -> RT-1 (``rt_1_tf_trained_for_000400120``)
  * WidowX / Bridge (4 tasks) -> Octo-small (``policy_setup='widowx_bridge'``)

Each task runs in its own subprocess (SAPIEN keeps a process-global renderer).

Run:  JAX_PLATFORMS=cpu VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      python scripts/render_policy_gallery.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

RT1_CKPT = os.environ.get("RT1_CKPT", "/home/ghr/projects/SimplerEnv/checkpoints/rt_1_tf_trained_for_000400120")
OUT = "/tmp/policy_gallery"
MAX_SEEDS = 8

GOOGLE = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
]
WIDOWX = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]
ALL = GOOGLE + WIDOWX
CAM = {"google": "overhead_camera", "widowx": "3rd_view_camera"}


def _family(task):
    return "widowx" if task.startswith("widowx") else "google"


def _make_policy(task):
    fam = _family(task)
    if fam == "google":
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        return RT1Inference(saved_model_path=RT1_CKPT, policy_setup="google_robot")
    from simpler_env.policies.octo.octo_model import OctoInference

    return OctoInference(model_type="octo-base", policy_setup="widowx_bridge", action_scale=1.0, init_rng=0)


def _img(obs, cam):
    return np.asarray(obs["image"][cam]["rgb"], dtype=np.uint8)


def run_one(task):
    from roboverse_pack.tasks.simpler_env._metasim.registry import TASK_MAP

    fam = _family(task)
    cam = CAM[fam]
    cls, kw = TASK_MAP[task]
    env = cls(**kw)
    policy = _make_policy(task)

    max_steps = int(getattr(env, "max_episode_steps", 120))  # each task's own horizon
    best = None  # (frames, success, steps, seed)
    for seed in range(MAX_SEEDS):
        obs, info = env.reset(seed=seed)
        instr = env.get_language_instruction()
        policy.reset(instr)
        frames = [_img(obs, cam)]
        succ = False
        steps = 0
        for t in range(max_steps):
            _, act = policy.step(frames[-1], instr)
            a = np.concatenate([act["world_vector"], act["rot_axangle"], act["gripper"]]).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            frames.append(_img(obs, cam))
            steps = t + 1
            succ = succ or bool(info.get("success", False))
            if succ or term or trunc:
                break
        cand = (np.stack(frames), succ, steps, seed)
        if best is None or (succ and not best[1]) or (succ and best[1] and steps < best[2]):
            best = cand
        if succ:
            break

    frames, succ, steps, seed = best
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{task}.npz", frames=frames, success=succ, steps=steps, seed=seed)
    print(f"{task}: success={succ} steps={steps} seed={seed} frames={len(frames)}")


def compose():
    import imageio

    results = {}
    for task in ALL:
        d = np.load(f"{OUT}/{task}.npz")
        frames = d["frames"]
        out = [f for f in frames] + [frames[-1]] * 4  # hold last frame
        imageio.mimsave(f"{OUT}/{task}.mp4", out, fps=7)
        results[task] = {"success": bool(d["success"]), "steps": int(d["steps"]), "seed": int(d["seed"])}
        print(f"OK {task} -> {OUT}/{task}.mp4 success={results[task]['success']} steps={results[task]['steps']}")
    n_ok = sum(v["success"] for v in results.values())
    summary = {"n_success": n_ok, "n": len(results), "per_task": results}
    with open(f"{OUT}/policy_gallery_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== policy success: {n_ok}/{len(results)} tasks ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["one", "compose", "all"], default="all")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.mode == "one":
        return run_one(a.task)
    if a.mode == "compose":
        return compose()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        TF_CPP_MIN_LOG_LEVEL="3",
        LOGURU_LEVEL="WARNING",
        VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json",
    )
    for task in ALL:
        r = subprocess.run([sys.executable, __file__, "--mode", "one", "--task", task], env=env, check=False)
        if r.returncode != 0:
            print(f"FAIL {task}")
    subprocess.run([sys.executable, __file__, "--mode", "compose"], env=env, check=False)


if __name__ == "__main__":
    main()
