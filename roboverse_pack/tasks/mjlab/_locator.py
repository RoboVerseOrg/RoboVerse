"""Resolve mjlab MJCF assets, preferring a local clone but falling back to HF.

mjlab is an external repo we don't vendor. Tasks here reference its MJCF files
by their *mjlab-internal* relpath (the path under ``src/mjlab/``), e.g.
``asset_zoo/robots/unitree_g1/xmls/g1.xml`` or ``tasks/cartpole/cartpole.xml``.

Resolution order (see :func:`mjlab_asset`):
  1. A local mjlab clone — ``MJLAB_REPO`` env var or ``~/projects/mjlab``.
     Used for development so the canonical mjlab bytes drive parity checks.
  2. HuggingFace ``RoboVerseOrg/roboverse_data`` under ``robots/mjlab/<relpath>``.
     This is what makes a fresh checkout work without cloning mjlab — the XML
     and every mesh it references are downloaded on demand.
"""

from __future__ import annotations

import os
from pathlib import Path

# Local mirror prefix inside the roboverse_data HF dataset. The on-HF layout
# mirrors mjlab's own ``src/mjlab/`` tree so resolution is a pure prefix swap.
_HF_LOCAL_DIR = "roboverse_data"
_HF_MJLAB_PREFIX = "robots/mjlab"

# Well-known mjlab-internal asset relpaths (under ``src/mjlab/``), shared by the
# task modules that load them via :func:`mjlab_asset`.
_CARTPOLE_XML = "tasks/cartpole/cartpole.xml"


def mjlab_repo_path() -> Path:
    """Locate a local mjlab clone, or raise if none is present."""
    env = os.environ.get("MJLAB_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MJLAB_REPO points at missing path: {p}")
        return p
    candidates = [
        Path.home() / "projects" / "mjlab",
    ]
    for c in candidates:
        if (c / "src" / "mjlab").exists():
            return c
    raise FileNotFoundError(
        "Could not locate mjlab. Clone https://github.com/mujocolab/mjlab into ~/projects/mjlab or set MJLAB_REPO."
    )


def hf_mjlab_path(relpath: str) -> str:
    """Local roboverse_data path that mirrors mjlab-internal ``relpath``."""
    return os.path.join(_HF_LOCAL_DIR, _HF_MJLAB_PREFIX, relpath)


def mjlab_asset(relpath: str) -> str:
    """Resolve an mjlab MJCF (and its meshes) to a concrete local path.

    Args:
        relpath: mjlab-internal path under ``src/mjlab/``, e.g.
            ``asset_zoo/robots/unitree_g1/xmls/g1.xml``.

    Returns:
        Absolute/relative filesystem path that exists locally. Prefers a local
        mjlab clone; otherwise downloads the XML and its referenced meshes from
        ``RoboVerseOrg/roboverse_data`` and returns the roboverse_data path.
    """
    # 1. Local mjlab clone (development fast path; canonical bytes for parity).
    try:
        p = mjlab_repo_path() / "src" / "mjlab" / relpath
        if p.exists():
            return str(p)
    except FileNotFoundError:
        pass

    # 2. HuggingFace fallback — download XML + meshes into roboverse_data/.
    local = hf_mjlab_path(relpath)
    if not os.path.exists(local):
        # Lazy import: hf_util pulls in heavy deps and we only need it on a
        # cold (clone-less) machine.
        from metasim.utils.hf_util import check_and_download_recursive

        check_and_download_recursive([local])
    return local
