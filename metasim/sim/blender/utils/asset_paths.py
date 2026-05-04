"""Lookup + on-demand download of Blender-side rendering assets.

Two assets the Blender backend can use for photoreal augmentation:

* HDRI library — equirectangular .hdr files used as image-based world lighting.
* Material library — a single .blend file (PBR materials) appended into the
  current scene to randomise object/ground appearance.

Both are large. We don't ship them in the repo. Instead we look in a small set
of well-known local paths first, and otherwise fetch from the project's
HuggingFace dataset under ``RoboVerseOrg/roboverse_data/blender/``. Returns
None if the asset cannot be located and HF download is disabled or fails — the
demo scripts treat that as "skip this augmentation".

Layout on HF (mirrors what we look for locally)::

    blender/
        envmap_lib/                 # 74 x *.hdr (~318 MB)
        envmap_sample/              # 4-6 x *.hdr (~10 MB, lightweight default)
        material_lib_v2.blend       # ~1.5 GB
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger as log

HF_REPO_ID = "RoboVerseOrg/roboverse_data"
HF_REPO_TYPE = "dataset"

LOCAL_ROOT = Path("roboverse_data/blender")
HF_BLENDER_PREFIX = "blender"

# Local search paths, tried in order. The Open6DOR copies are a legacy layout
# from before these assets were promoted to the roboverse_data HF repo.
HDRI_LOCAL_CANDIDATES: tuple[Path, ...] = (
    LOCAL_ROOT / "envmap_lib",
    LOCAL_ROOT / "envmap_sample",
    Path("third_party/Open6DOR/assets/envmap_lib"),
    Path("assets/envmap_lib"),
)
MATERIAL_LIB_LOCAL_CANDIDATES: tuple[Path, ...] = (
    LOCAL_ROOT / "material_lib_v2.blend",
    Path("third_party/Open6DOR/assets/material_lib_v2.blend"),
    Path("assets/material_lib_v2.blend"),
)


def find_local_hdri_dir() -> Path | None:
    """Return the first existing local HDRI directory containing .hdr files."""
    for cand in HDRI_LOCAL_CANDIDATES:
        if cand.is_dir() and any(cand.glob("*.hdr")):
            return cand
    return None


def find_local_material_library() -> Path | None:
    """Return the first existing local material .blend file."""
    for cand in MATERIAL_LIB_LOCAL_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def _download_from_hf(remote_relpath: str, local_path: Path) -> Path | None:
    """Download a single file from the roboverse_data HF dataset.

    ``remote_relpath`` is the path under the repo root (e.g.
    ``blender/material_lib_v2.blend``). The file is written to
    ``roboverse_data/<remote_relpath>`` so it shows up in the same place
    later runs look first.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning("huggingface_hub not installed — cannot fetch {}", remote_relpath)
        return None
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=remote_relpath,
            local_dir="roboverse_data",
        )
        return Path(downloaded)
    except Exception as e:
        log.warning("HF download failed for {}: {}", remote_relpath, e)
        return None


def _list_hf_dir(remote_dir: str) -> list[str]:
    """List file paths under a remote dataset directory, returning [] on failure."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    except Exception as e:
        log.warning("HF list_repo_files failed: {}", e)
        return []
    prefix = remote_dir.rstrip("/") + "/"
    return [f for f in files if f.startswith(prefix)]


def get_hdri_dir(download: bool = True, sample_only: bool = True) -> Path | None:
    """Return a directory containing at least one HDRI, downloading if needed.

    Args:
        download: if True, fetch from HF when no local copy is found.
        sample_only: pull the small ``envmap_sample`` subset (a handful of
            .hdr files, ~10 MB) instead of the full 318 MB ``envmap_lib``.
    """
    local = find_local_hdri_dir()
    if local is not None:
        return local
    if not download:
        return None

    remote_dir = f"{HF_BLENDER_PREFIX}/{'envmap_sample' if sample_only else 'envmap_lib'}"
    target = LOCAL_ROOT / ("envmap_sample" if sample_only else "envmap_lib")
    log.info(f"fetching HDRIs from HF: {HF_REPO_ID}/{remote_dir}")
    files = [f for f in _list_hf_dir(remote_dir) if f.lower().endswith(".hdr")]
    if not files:
        log.warning(f"no HDRIs at HF://{remote_dir}; skipping")
        return None
    for relpath in files:
        local_file = LOCAL_ROOT.parent / relpath  # roboverse_data/<relpath>
        if local_file.exists():
            continue
        _download_from_hf(relpath, local_file)
    return target if target.is_dir() and any(target.glob("*.hdr")) else None


def get_material_library(download: bool = True) -> Path | None:
    """Return the path to ``material_lib_v2.blend``, downloading if needed.

    The file is ~1.5 GB. Callers that don't truly need it should pass
    ``download=False`` and fall back to procedural materials.
    """
    local = find_local_material_library()
    if local is not None:
        return local
    if not download:
        return None
    remote = f"{HF_BLENDER_PREFIX}/material_lib_v2.blend"
    target = LOCAL_ROOT / "material_lib_v2.blend"
    log.info(f"fetching material library from HF (~1.5 GB): {HF_REPO_ID}/{remote}")
    return _download_from_hf(remote, target)


__all__ = [
    "HF_REPO_ID",
    "find_local_hdri_dir",
    "find_local_material_library",
    "get_hdri_dir",
    "get_material_library",
]
