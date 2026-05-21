"""ffmpeg hstack two mp4s with labels into a side-by-side comparison."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def stitch(
    left_mp4: Path,
    right_mp4: Path,
    out_mp4: Path,
    *,
    left_label: str = "mjlab native (mujoco-warp)",
    right_label: str = "MetaSim/MuJoCo (action replay)",
    height: int = 360,
) -> int:
    if not shutil.which("ffmpeg"):
        print("[stitch] ffmpeg not found in PATH", file=sys.stderr)
        return 1
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Scale both to fixed height keeping aspect, add label, hstack.
    # Use drawtext with a fallback font path.
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    font = next((f for f in font_candidates if Path(f).exists()), None)
    if font is None:
        # fall back: hstack without labels
        filter_complex = f"[0:v]scale=-2:{height}[l];[1:v]scale=-2:{height}[r];[l][r]hstack=inputs=2"
    else:
        filter_complex = (
            f"[0:v]scale=-2:{height},drawtext=fontfile={font}:text='{left_label}':x=10:y=10:fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=6[l];"
            f"[1:v]scale=-2:{height},drawtext=fontfile={font}:text='{right_label}':x=10:y=10:fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:boxborderw=6[r];"
            f"[l][r]hstack=inputs=2"
        )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left_mp4),
        "-i",
        str(right_mp4),
        "-filter_complex",
        filter_complex,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-r",
        "30",
        str(out_mp4),
    ]
    print(f"[stitch] {out_mp4}")
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        print("[stitch] ffmpeg failed:", res.stderr[-1000:], file=sys.stderr)
        return res.returncode
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--left-label", default="mjlab native (mujoco-warp)")
    p.add_argument("--right-label", default="MetaSim/MuJoCo (action replay)")
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args(argv)
    return stitch(
        Path(args.left),
        Path(args.right),
        Path(args.out),
        left_label=args.left_label,
        right_label=args.right_label,
        height=args.height,
    )


if __name__ == "__main__":
    sys.exit(main())
