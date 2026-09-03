"""End-to-end SimplerEnv passthrough demo + visualization.

Drives a SimplerEnv task *through the RoboVerse passthrough*
(``gymnasium.make("SimplerEnv/<task>")``) — i.e. the full integration path, not
the upstream env directly — and writes an MP4 of the rendered rollout plus the
camera-image observation stream. This is both the end-to-end integration smoke
test and the visualization used in the research_report page.

For each task it also (optionally) runs the same seeded rollout on the native
``simpler_env.make`` env and stacks the two side by side, so the video itself is
visual proof of 1:1 (left = native SimplerEnv, right = RoboVerse passthrough).

Usage::

    python tools/parity/render_simpler_env.py --tasks google_robot_pick_coke_can \
        widowx_put_eggplant_in_basket --steps 60 --out /tmp/simpler_videos

Requires the dedicated ``simpler`` conda env (SimplerEnv + SAPIEN + a GPU/Vulkan).
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _camera_name(env) -> str:
    uid = getattr(getattr(env, "unwrapped", env), "robot_uid", "") or ""
    if "google_robot" in uid:
        return "overhead_camera"
    if "widowx" in uid:
        return "3rd_view_camera"
    return ""


def _as_rgb(x):
    """Return x as an HxWx3 uint8 array, or None if it isn't an image."""
    try:
        a = np.asarray(x)
    except Exception:
        return None
    if a.ndim == 3 and a.shape[-1] == 3 and np.issubdtype(a.dtype, np.integer):
        return a.astype(np.uint8)
    return None


def _image_from_obs(env, obs) -> np.ndarray | None:
    """Extract an RGB frame (HxWx3 uint8) from a SimplerEnv obs dict."""
    try:
        img = obs["image"]
        cam = _camera_name(env)
        order = ([cam] if cam and cam in img else []) + list(img.keys())
        for c in order:
            rgb = _as_rgb(img[c].get("rgb"))
            if rgb is not None:
                return rgb
    except Exception:
        return None
    return None


def _rollout_frames(env, actions, seed):
    obs, _ = env.reset(seed=seed)
    frames = []
    f = _image_from_obs(env, obs)
    if f is not None:
        frames.append(f)
    for a in actions:
        out = env.step(a)
        obs = out[0]
        f = _image_from_obs(env, obs)
        if f is not None:
            frames.append(f)
    return frames


def _save_mp4(frames, path, fps=10):
    import imageio

    if not frames:
        return False
    imageio.mimsave(path, [np.asarray(f).astype(np.uint8) for f in frames], fps=fps)
    return True


def render_task(task_name, *, steps, seed, out_dir, side_by_side):
    import gymnasium as gym
    import simpler_env

    from roboverse_pack.tasks.simpler_env import register_simpler_env_passthrough

    register_simpler_env_passthrough()

    wrapped = gym.make(f"SimplerEnv/{task_name}")
    rng = np.random.RandomState(seed)
    space = wrapped.action_space
    actions = [rng.uniform(space.low, space.high).astype(space.dtype) for _ in range(steps)]

    fw = _rollout_frames(wrapped, actions, seed)
    wrapped.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.mp4")

    if side_by_side:
        native = simpler_env.make(task_name)
        fn = _rollout_frames(native, actions, seed)
        native.close()
        n = min(len(fn), len(fw))
        combo = []
        for i in range(n):
            a = np.asarray(fn[i])
            b = np.asarray(fw[i])
            if a.shape == b.shape:
                combo.append(np.concatenate([a, b], axis=1))
        if combo:
            ok = _save_mp4(combo, out_path)
            return out_path if ok else None

    ok = _save_mp4(fw, out_path)
    return out_path if ok else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=["google_robot_pick_coke_can"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="/tmp/simpler_videos")
    parser.add_argument("--no-side-by-side", action="store_true", help="render only the passthrough env")
    args = parser.parse_args()

    import simpler_env

    tasks = list(simpler_env.ENVIRONMENTS) if args.all else args.tasks
    for t in tasks:
        try:
            p = render_task(
                t, steps=args.steps, seed=args.seed, out_dir=args.out, side_by_side=not args.no_side_by_side
            )
            print(f"[OK]   {t}: {p}")
        except Exception as exc:
            print(f"[FAIL] {t}: {exc!r}")


if __name__ == "__main__":
    main()
