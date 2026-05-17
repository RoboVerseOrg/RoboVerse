"""Replay an MJCF trajectory and write a video via mujoco.Renderer.

Offscreen rendering uses the EGL backend (configured via the MUJOCO_GL env
var by the caller). If a renderer cannot be created we write a placeholder
PNG so the report doesn't break.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from .raw_rollout import RolloutResult


def replay_to_video(
    xml_path: str | Path,
    rollout: RolloutResult,
    out_path: Path,
    *,
    width: int = 480,
    height: int = 360,
    camera: str | int = -1,
    frame_stride: int = 1,
    fps: int | None = None,
) -> str | None:
    """Replay qpos/qvel onto a fresh model and write an mp4. Returns out path or None."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
    except Exception as e:  # pragma: no cover — depends on local GL env
        print(f"[mjlab_integration] Renderer init failed for {xml_path}: {e}")
        _write_placeholder(out_path.with_suffix(".png"), str(e))
        return None

    if fps is None:
        # Default to a soft 30fps target relative to the control rate.
        ctrl_dt = rollout.dt * rollout.decimation
        fps = max(1, round(1.0 / max(ctrl_dt, 1e-3)))

    try:
        import imageio
    except Exception as e:  # pragma: no cover
        print(f"[mjlab_integration] imageio missing: {e}")
        return None

    frames = []
    for k in range(0, rollout.qpos.shape[0], frame_stride):
        data.qpos[:] = rollout.qpos[k]
        data.qvel[:] = rollout.qvel[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

    try:
        imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=7)
    except Exception:
        # fall back to GIF if libx264 not available
        gif_path = out_path.with_suffix(".gif")
        imageio.mimwrite(gif_path, frames, fps=fps)
        out_path = gif_path

    return str(out_path)


def _write_placeholder(png_path: Path, message: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.text(0.5, 0.5, "Renderer unavailable:\n" + message, ha="center", va="center", wrap=True, fontsize=10)
    ax.set_axis_off()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def configure_offscreen_gl() -> None:
    """Set MUJOCO_GL=egl (or osmesa fallback) for headless rendering."""
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    # If EGL fails we'll retry with osmesa silently in the caller.


def _render_frames(
    xml_path: str | Path,
    rollout: RolloutResult,
    *,
    width: int,
    height: int,
    camera: str | int,
    frame_stride: int,
) -> tuple[list, int]:
    """Replay `rollout.qpos/qvel` onto a fresh model from `xml_path` and return frames + nq."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)

    raw_nq = int(rollout.qpos.shape[1])
    model_nq = int(model.nq)

    frames: list = []
    for k in range(0, rollout.qpos.shape[0], frame_stride):
        q = rollout.qpos[k]
        v = rollout.qvel[k]
        # qpos shapes can differ if the metasim model added wrapper qpos
        # slots (rare — wrappers usually have no joint). Truncate or pad so
        # we never crash; visual quality may suffer but the report still
        # gets a frame instead of a blank.
        if q.shape[0] >= model_nq:
            data.qpos[:] = q[:model_nq]
        else:
            data.qpos[: q.shape[0]] = q
        nv = int(model.nv)
        if v.shape[0] >= nv:
            data.qvel[:] = v[:nv]
        else:
            data.qvel[: v.shape[0]] = v
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    return frames, raw_nq


def side_by_side_video(
    xml_path: str | Path,
    raw_rollout: RolloutResult,
    meta_rollout: RolloutResult,
    out_path: Path,
    *,
    width: int = 360,
    height: int = 270,
    camera: str | int = -1,
    frame_stride: int = 1,
    fps: int | None = None,
    labels: tuple[str, str] = ("raw mujoco", "MetaSim full"),
) -> str | None:
    """Render both rollouts on the same MJCF and concat into a side-by-side mp4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
        import numpy as np
    except Exception as e:
        print(f"[mjlab_integration] imageio/numpy missing: {e}")
        return None

    try:
        raw_frames, _ = _render_frames(
            xml_path, raw_rollout, width=width, height=height, camera=camera, frame_stride=frame_stride
        )
        meta_frames, _ = _render_frames(
            xml_path, meta_rollout, width=width, height=height, camera=camera, frame_stride=frame_stride
        )
    except Exception as e:
        print(f"[mjlab_integration] render failed: {e}")
        _write_placeholder(out_path.with_suffix(".png"), str(e))
        return None

    n_frames = min(len(raw_frames), len(meta_frames))
    if n_frames == 0:
        return None

    # Stamp the two halves with a tiny label so users can tell them apart.
    try:
        from PIL import Image, ImageDraw

        def _label(arr, text):
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, len(text) * 7 + 8, 18], fill=(0, 0, 0, 200))
            draw.text((4, 2), text, fill=(255, 255, 255))
            return np.asarray(img)

    except Exception:

        def _label(arr, text):
            return arr

    combined = []
    for k in range(n_frames):
        a = _label(raw_frames[k], labels[0])
        b = _label(meta_frames[k], labels[1])
        # Pad the shorter axis with black if shapes don't match (size-locked already).
        h = max(a.shape[0], b.shape[0])
        if a.shape[0] != h:
            pad = np.zeros((h - a.shape[0], a.shape[1], a.shape[2]), dtype=a.dtype)
            a = np.vstack([a, pad])
        if b.shape[0] != h:
            pad = np.zeros((h - b.shape[0], b.shape[1], b.shape[2]), dtype=b.dtype)
            b = np.vstack([b, pad])
        sep = np.zeros((h, 4, a.shape[2]), dtype=a.dtype)
        combined.append(np.hstack([a, sep, b]))

    if fps is None:
        ctrl_dt = raw_rollout.dt * raw_rollout.decimation * max(1, frame_stride)
        fps = max(1, round(1.0 / max(ctrl_dt, 1e-3)))

    try:
        imageio.mimwrite(out_path, combined, fps=fps, codec="libx264", quality=7)
    except Exception:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimwrite(gif_path, combined, fps=fps)
        out_path = gif_path

    return str(out_path)
