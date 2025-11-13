#!/usr/bin/env python3
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from textwrap import dedent

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_DIR = REPO_ROOT / "third_party" / "InteriorAgent"
ASSET_DIR = REPO_ROOT / "roboverse_pack" / "asset"
ROBOVERSE_REPO_ID = "RoboVerseOrg/roboverse_data"
HF_SCENES_PATTERN = "scenes/kujiale/*.usda"

logger = logging.getLogger(__name__)

LICENSE_NOTICE = dedent(
    """
    NOTICE: The Kujiale interior scene assets are sourced from the InteriorAgent dataset.
    These files are not redistributed in this repository. Please obtain them directly
    and comply with the InteriorAgent Terms of Use before downloading or citing them:
    https://kloudsim-usa-cos.kujiale.com/InteriorAgent/InteriorAgent_Terms_of_Use.pdf
    """
).strip()


def download_roboverse_usd() -> None:
    """Download RoboVerse-curated Kujiale USD scenes from Hugging Face."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = ASSET_DIR / ".hf_roboverse_cache"
    snapshot_download(
        repo_id=ROBOVERSE_REPO_ID,
        repo_type="dataset",
        local_dir=str(temp_dir),
        allow_patterns=HF_SCENES_PATTERN,
        local_dir_use_symlinks=False,
    )

    source_dir = temp_dir / "scenes" / "kujiale"
    for usd_file in sorted(source_dir.glob("*.usda")):
        shutil.copy2(usd_file, ASSET_DIR / usd_file.name)

    shutil.rmtree(temp_dir, ignore_errors=True)


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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for line in LICENSE_NOTICE.splitlines():
        logger.info(line)
    download_roboverse_usd()
    download_interior_agent()
    copy_usd_assets()


if __name__ == "__main__":
    main()
