"""Optional Kit entry point for Blender USD material overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

from .overlay import generate_blender_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Blender material overlay without requiring Kit imports.")
    parser.add_argument("--input", help="Input USD stage.")
    parser.add_argument("--overlay", help="Output overlay USD layer.")
    parser.add_argument("--root", help="Output root USD layer.")
    parser.add_argument("--texture-cache", help="Texture cache/report directory.")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(argv)

    if not (args.input and args.overlay and args.root and args.texture_cache):
        print("Kit MDL bake is not enabled; provide --input, --overlay, --root, and --texture-cache to generate the MVP overlay.")
        return 0

    generate_blender_overlay(
        Path(args.input),
        Path(args.overlay),
        Path(args.root),
        Path(args.texture_cache),
        resolution=args.resolution,
        samples=args.samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

