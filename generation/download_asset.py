from __future__ import annotations

import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, ".."))

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from glob import glob

import tyro
from huggingface_hub import list_repo_files, snapshot_download
from rich.console import Console
from rich.table import Table

from generation.enums import AssetType
from generation.load_asset import cvt_embodiedgen_asset_to_anysim

console = Console()


@dataclass
class EmbodiedGenDownloadConfig:
    """Configuration for downloading EmbodiedGen assets."""

    repo_id = "HorizonRobotics/EmbodiedGenData"
    download_local_dir: str = "roboverse_data/assets/EmbodiedGenData"
    target_type: str = "dataset/basic_furniture/table"
    uuid: str = "*"
    download_num: int | None = None


def get_real_height(urdf_path: str) -> float:
    """Get the real height of an asset from its URDF file."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    elem = root.find(".//extra_info/real_height")

    return float(elem.text)


if __name__ == "__main__":
    cfg = tyro.cli(EmbodiedGenDownloadConfig)

    target_patterns = cfg.target_type if cfg.uuid == "*" else f"{cfg.target_type}/{cfg.uuid}"
    all_files = list_repo_files(repo_id=cfg.repo_id, repo_type="dataset")
    keep_files = [f for f in all_files if f.startswith(target_patterns)]
    uuids = sorted(set([f.split("/")[3] for f in keep_files]))
    uuids = uuids[: cfg.download_num] if cfg.download_num is not None else uuids
    allow_patterns = [f"{cfg.target_type}/{uid}/*" for uid in uuids]

    snapshot_download(
        repo_id=cfg.repo_id,
        repo_type="dataset",
        local_dir=cfg.download_local_dir,
        allow_patterns=allow_patterns,
    )

    urdf_files = glob(f"{cfg.download_local_dir}/{cfg.target_type}/*/*.urdf")

    table = Table(title="Downloaded Assets & Real Heights")
    table.add_column("Asset Path", style="cyan")
    table.add_column("Real Height (m)", style="green")

    for path in urdf_files:
        height = get_real_height(path)
        table.add_row(path, f"{height:.3f}")

    dst_asset_path = cvt_embodiedgen_asset_to_anysim(
        urdf_files=urdf_files,
        target_dirs=[f"{os.path.dirname(path)}/mjcf" for path in urdf_files],
        target_type=AssetType.MJCF,
        source_type=AssetType.MESH,
    )

    from isaaclab.app import AppLauncher

    launch_args = dict(
        headless=True,
        no_splash=True,
        fast_shutdown=True,
        disable_gpu=True,
    )
    app_launcher = AppLauncher(launch_args)
    simulation_app = app_launcher.app

    dst_asset_path = cvt_embodiedgen_asset_to_anysim(
        urdf_files=urdf_files,
        target_dirs=[f"{os.path.dirname(path)}/usd" for path in urdf_files],
        target_type=AssetType.USD,
        source_type=AssetType.MESH,
        simulation_app=simulation_app,
    )

    console.print(table)
    console.print(f"Downloaded {len(uuids)} assets to {cfg.download_local_dir}/{cfg.target_type}")

    simulation_app.close()
