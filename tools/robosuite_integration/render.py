"""Offscreen rendering helpers for parity videos."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def render_qpos_sequence(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    *,
    camera: str = "frontview",
    width: int = 384,
    height: int = 384,
) -> list[np.ndarray]:
    """Render an RGB frame for each qpos by forward-kinematics only (no stepping)."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = (
        camera
        if any(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) == camera for i in range(model.ncam))
        else 0
    )
    frames = []
    for qpos in qpos_seq:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption banner at the top of a frame (PIL if available, else plain)."""
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 18], fill=(20, 20, 28))
        draw.text((4, 3), text, fill=(235, 235, 245))
        return np.asarray(img)
    except Exception:
        return frame


def _banner(frame: np.ndarray, text: str, rgb: tuple[int, int, int]) -> np.ndarray:
    """Draw a status banner across the bottom of a frame."""
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        h = img.height
        draw.rectangle([0, h - 20, img.width, h], fill=rgb)
        draw.text((6, h - 17), text, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return frame


def side_by_side(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    out_path: str,
    *,
    label_a: str = "robosuite (native)",
    label_b: str = "MetaSim MujocoHandler",
    status: list[bool] | None = None,
    fps: int = 20,
) -> str:
    """Stitch two equal-length frame lists side-by-side and write an mp4.

    ``status`` (optional, per-frame bool) draws a green "TASK COMPLETE" / grey
    "in progress" banner so the video objectively shows task success.
    """
    import imageio

    n = min(len(frames_a), len(frames_b))
    sep = np.full((frames_a[0].shape[0], 4, 3), 60, dtype=np.uint8)
    combined = []
    for i in range(n):
        a, b = _label(frames_a[i], label_a), _label(frames_b[i], label_b)
        if status is not None:
            done = status[i]
            txt = "TASK COMPLETE  (_check_success = True)" if done else "in progress"
            rgb = (16, 130, 70) if done else (70, 70, 78)
            a, b = _banner(a, txt, rgb), _banner(b, txt, rgb)
        combined.append(np.hstack([a, sep, b]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimwrite(out_path, combined, fps=fps, codec="libx264", quality=8)
    return out_path
