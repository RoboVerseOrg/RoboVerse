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
