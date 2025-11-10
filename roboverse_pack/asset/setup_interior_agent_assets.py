#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_DIR = REPO_ROOT / "third_party" / "InteriorAgent"
ASSET_DIR = REPO_ROOT / "roboverse_pack" / "asset"


def download_interior_agent() -> None:
    """Download the InteriorAgent dataset from Hugging Face."""
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="spatialverse/InteriorAgent",
        repo_type="dataset",
        local_dir=str(THIRD_PARTY_DIR),
        local_dir_use_symlinks=False,
    )


def copy_usd_assets() -> None:
    """Copy the three-digit USD scenes into the asset directory."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    for scene_dir in sorted(THIRD_PARTY_DIR.glob("kujiale_*")):
        if not scene_dir.is_dir():
            continue

        for usd_file in scene_dir.glob("*.usda"):
            stem = usd_file.stem
            if stem.isdigit() and len(stem) == 3:
                shutil.copy2(usd_file, ASSET_DIR / usd_file.name)


def main() -> None:
    """Download and arrange InteriorAgent assets for RoboVerse."""
    download_interior_agent()
    copy_usd_assets()


if __name__ == "__main__":
    main()
