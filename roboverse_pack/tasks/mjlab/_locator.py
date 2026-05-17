"""Resolve mjlab asset paths from a sibling clone or `MJLAB_REPO` env var.

mjlab is an external repo we don't vendor — tasks here just point at its
MJCF files. The expected layout matches `tools/mjlab_integration/inventory.py`.
"""

from __future__ import annotations

import os
from pathlib import Path


def mjlab_repo_path() -> Path:
    env = os.environ.get("MJLAB_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MJLAB_REPO points at missing path: {p}")
        return p
    candidates = [
        Path.home() / "projects" / "mjlab",
        Path("/home/ghr/projects/mjlab"),
    ]
    for c in candidates:
        if (c / "src" / "mjlab").exists():
            return c
    raise FileNotFoundError(
        "Could not locate mjlab. Clone https://github.com/mujocolab/mjlab into ~/projects/mjlab or set MJLAB_REPO."
    )


def mjlab_asset(relpath: str) -> str:
    return str(mjlab_repo_path() / "src" / "mjlab" / relpath)
