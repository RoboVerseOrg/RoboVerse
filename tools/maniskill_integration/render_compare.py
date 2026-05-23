"""Render side-by-side comparison: ManiSkill PickCube-v1 vs MetaSim pick_cube_v1.

This is the visual companion to the parity table in the integration
report. Both halves are rendered offscreen (ManiSkill via its built-in
`render_mode='rgb_array'`; MetaSim via a Sapien camera added manually
to the scene), then concatenated.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

from .maniskill_rollout import rollout_maniskill

_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")


def _label_frame(img, text):
    try:
        from PIL import Image, ImageDraw

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.5 else img.astype(np.uint8)
        pil = Image.fromarray(img[..., :3])
        draw = ImageDraw.Draw(pil)
        w = len(text) * 7 + 8
        draw.rectangle([0, 0, w, 18], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(255, 255, 255))
        return np.asarray(pil)
    except Exception:
        return img[..., :3] if img.shape[-1] == 4 else img


def rollout_metasim_pick_cube_v1(n_steps: int = 30, seed: int = 0, render_size=(320, 240)):
    """Step MetaSim's `maniskill.pick_cube_v1` task and grab rgb frames."""
    import sapien

    import roboverse_pack.tasks.maniskill  # noqa: F401  register tasks
    from metasim.task.registry import get_task_class

    cls = get_task_class("maniskill.pick_cube_v1")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []

    env = cls(sc)
    env.reset()
    scene = env.handler.scene

    # Add a small offscreen camera so we can grab frames. Pose mirrors a
    # typical PickCube look-at from front-right.
    cam = scene.add_camera("rgb_compare", width=render_size[0], height=render_size[1], fovy=1.0, near=0.05, far=10)
    # Look down at the workspace centered roughly at the table.
    cam.set_local_pose(sapien.Pose([0.4, -0.6, 0.55], [0.65, 0.0, 0.30, 0.70]))

    frames = []
    for _ in range(n_steps):
        scene.update_render()
        cam.take_picture()
        rgb = cam.get_picture("Color")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 and rgb.max() <= 1.5:
            rgb = (rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb[..., :3].astype(np.uint8)
        frames.append(rgb)
        env.handler.simulate()

    env.handler.close()
    return frames


def side_by_side(maniskill_frames, metasim_frames, out_mp4: Path, fps: int = 20):
    """Concat frame-by-frame; pad shorter side to the same length."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
    except Exception:
        print("[maniskill render_compare] imageio missing")
        return None

    n = min(len(maniskill_frames), len(metasim_frames))
    if n == 0:
        return None

    combined = []
    for k in range(n):
        a = _label_frame(maniskill_frames[k], "ManiSkill PickCube-v1")
        b = _label_frame(metasim_frames[k], "MetaSim pick_cube_v1")
        # Pad to matching height.
        h = max(a.shape[0], b.shape[0])
        if a.shape[0] != h:
            pad = np.zeros((h - a.shape[0], a.shape[1], a.shape[2]), dtype=a.dtype)
            a = np.vstack([a, pad])
        if b.shape[0] != h:
            pad = np.zeros((h - b.shape[0], b.shape[1], b.shape[2]), dtype=b.dtype)
            b = np.vstack([b, pad])
        sep = np.zeros((h, 4, 3), dtype=a.dtype)
        combined.append(np.hstack([a, sep, b]))

    try:
        imageio.mimwrite(out_mp4, combined, fps=fps, codec="libx264", quality=7)
    except Exception:
        gif = out_mp4.with_suffix(".gif")
        imageio.mimwrite(gif, combined, fps=fps)
        out_mp4 = gif
    return str(out_mp4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPORTS, "maniskill_integration"))
    p.add_argument("--n-steps", type=int, default=30)
    args = p.parse_args(argv)
    out_dir = Path(args.out)

    # ManiSkill native — drive a small sinusoid so the arm visibly moves.
    import math

    print("[render_compare] running ManiSkill PickCube-v1...")

    def _sine(t, q):
        a = np.zeros(8, dtype=np.float32)
        # Slow oscillation on joint2 (shoulder) so the arm sweeps in/out.
        a[1] = 0.6 * math.sin(2 * math.pi * t / 30.0)
        # Slight bias to push arm down toward cube.
        a[3] = -0.2
        return a

    ms = rollout_maniskill(
        "PickCube-v1",
        n_steps=args.n_steps,
        seed=0,
        action_fn=_sine,
        render=True,
        render_size=(320, 240),
    )
    print(f"  ManiSkill: {len(ms.frames)} frames, reward Σ={ms.rewards.sum():.3f}")

    # MetaSim sidecar
    print("[render_compare] running MetaSim pick_cube_v1...")
    metasim_frames = rollout_metasim_pick_cube_v1(n_steps=args.n_steps, seed=0, render_size=(320, 240))
    print(f"  MetaSim:   {len(metasim_frames)} frames")

    out_mp4 = out_dir / "runs" / "maniskill.pick_cube_v1" / "side_by_side.mp4"
    written = side_by_side(ms.frames, metasim_frames, out_mp4, fps=20)
    print(f"[render_compare] wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
