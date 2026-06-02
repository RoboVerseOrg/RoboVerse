"""Resolve RoboTwin assets, preferring a local clone but falling back to a vendored mirror.

The RoboTwin integration must work **without the upstream RoboTwin checkout present**
(the "you could delete ~/projects/robotwin" requirement). Every asset a bridge references
-- object visual/collision meshes, URDF instances (``mobility.urdf`` + ``model_data*.json``),
and the ALOHA-AgileX embodiment -- is addressed by its *RoboTwin-internal relpath* (the path
under the RoboTwin repo root, e.g. ``assets/objects/020_hammer/visual/textured.obj`` or
``assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf``).

Resolution order (see :func:`robotwin_asset`):
  1. A local RoboTwin clone -- ``ROBOTWIN_ASSETS`` (pointing at the repo's ``assets/`` or the
     repo root) or ``~/projects/robotwin``. Used for development / fresh data collection, where
     the canonical RoboTwin bytes drive the bridge.
  2. The vendored mirror ``roboverse_data/robotwin/<relpath>`` -- populated by
     ``tools/robotwin_integration/migrate_assets.py`` and uploaded to the
     ``RoboVerseOrg/roboverse_data`` HuggingFace dataset. This is what makes a checkout work
     *after RoboTwin is deleted*: the file (and, on a cold machine, its HF download) lives
     entirely inside roboverse_data, exactly like the mjlab / menagerie locators.

A pure prefix swap keeps the two layouts identical: ``<robotwin_root>/<relpath>`` <->
``roboverse_data/robotwin/<relpath>``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Vendored-mirror prefix inside the roboverse_data HF dataset. The on-HF layout mirrors the
# RoboTwin repo tree so resolution is a pure prefix swap.
_HF_LOCAL_DIR = "roboverse_data"
_HF_ROBOTWIN_PREFIX = "robotwin"


def robotwin_clone_root() -> Path | None:
    """Locate a local RoboTwin clone root (the dir that contains ``assets/``), or None.

    ``ROBOTWIN_ASSETS`` may point at either the repo root or its ``assets/`` subdir; both are
    accepted. When ``ROBOTWIN_ASSETS`` is set it is **authoritative**: the implicit
    ``~/projects/robotwin`` fallback is skipped, so pointing it at a non-existent path is the
    supported way to force the vendored-mirror path (i.e. simulate "RoboTwin deleted").
    """
    env = os.environ.get("ROBOTWIN_ASSETS")
    if env:
        p = Path(env).expanduser()
        for c in (p, p.parent if p.name == "assets" else p):  # accept <root> or <root>/assets
            if (c / "assets").is_dir():
                return c
        return None  # explicit override that doesn't resolve -> no clone (use the mirror)
    default = Path.home() / "projects" / "robotwin"
    return default if (default / "assets").is_dir() else None


def hf_robotwin_path(relpath: str) -> str:
    """Local roboverse_data path mirroring the RoboTwin-internal ``relpath``."""
    return os.path.join(_HF_LOCAL_DIR, _HF_ROBOTWIN_PREFIX, relpath)


def robotwin_asset(relpath: str, *, download: bool = True) -> str:
    """Resolve a RoboTwin-internal ``relpath`` to a concrete filesystem path.

    Args:
        relpath: Path under the RoboTwin repo root, e.g.
            ``assets/objects/020_hammer/visual/textured.obj``.
        download: When True (default) and the file is absent from both the clone and the local
            mirror, pull it (and anything under the same dir) from ``RoboVerseOrg/roboverse_data``.

    Returns:
        A path that exists locally. Prefers a local RoboTwin clone (dev fast path / canonical
        bytes); otherwise the vendored ``roboverse_data/robotwin/<relpath>`` mirror.

    Raises:
        FileNotFoundError: if the asset is in neither location and cannot be downloaded.
    """
    relpath = relpath.lstrip("/")

    # Always return an ABSOLUTE path: callers wrap the mesh in a generated URDF and compute
    # relative mesh refs, and an absolute source avoids '../' traversal out of any sandbox.

    # 1. Local RoboTwin clone (development fast path; canonical bytes for parity).
    root = robotwin_clone_root()
    if root is not None:
        p = root / relpath
        if p.exists():
            return os.path.abspath(str(p))

    # 2. Vendored mirror (the RoboTwin-deletable path).
    local = hf_robotwin_path(relpath)
    if os.path.exists(local):
        return os.path.abspath(local)

    # 3. HuggingFace download into roboverse_data/ (cold, clone-less machine).
    if download:
        try:
            from metasim.utils.hf_util import check_and_download_recursive

            check_and_download_recursive([local])
        except Exception as e:
            raise FileNotFoundError(
                f"RoboTwin asset {relpath!r} not found in a clone, the roboverse_data mirror, "
                f"or downloadable from HuggingFace ({type(e).__name__}: {e}). Run "
                f"tools/robotwin_integration/migrate_assets.py against a RoboTwin clone to "
                f"populate roboverse_data/robotwin/."
            ) from e
        if os.path.exists(local):
            return os.path.abspath(local)

    raise FileNotFoundError(
        f"RoboTwin asset {relpath!r} not found (no clone at ROBOTWIN_ASSETS/~/projects/robotwin, "
        f"and roboverse_data/robotwin/ has no mirror). Populate it with "
        f"tools/robotwin_integration/migrate_assets.py."
    )
