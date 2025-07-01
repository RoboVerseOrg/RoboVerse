from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tyro

from metasim.utils.dex_util.constants import HandType, RobotName
from metasim.utils.dex_util.dataset import DexYCBVideoDataset
from metasim.utils.dex_util.hand_robot_viewer import RobotHandDatasetSAPIENViewer
from metasim.utils.dex_util.retargeting_config import RetargetingConfig

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def viz_retarget_only(
    dexycb_dir: Path,
    urdf_dir: Path,
    robot_names: tuple[RobotName, ...] | None,
    fps: int = 10,
    data_id: int = 4,
    headless: bool = False,
):
    # Set default URDF dir
    RetargetingConfig.set_default_urdf_dir(urdf_dir)

    # Load dataset
    dataset = DexYCBVideoDataset(dexycb_dir, hand_type="right")
    sampled_data = dataset[data_id]

    # Viewer
    if robot_names is None:
        robot_names = ()
    viewer = RobotHandDatasetSAPIENViewer(list(robot_names), HandType.right, headless=headless)
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps=fps)


def main(
    dexycb_dir: Path,
    urdf_dir: Path,
    robots: list[RobotName] | None = None,
    fps: int = 10,
    data_id: int = 0,
    headless: bool = False,
):
    """
    Render human hand + robot hand trajectories with object, and export video if headless.
    """
    dexycb_dir_path = Path(dexycb_dir).absolute()
    urdf_dir_path = Path(urdf_dir).absolute()
    if not dexycb_dir_path.exists():
        raise FileNotFoundError(f"DexYCB dir not found: {dexycb_dir_path}")
    if not urdf_dir_path.exists():
        raise FileNotFoundError(f"URDF dir not found: {urdf_dir_path}")

    logging.info(f"Using DexYCB dataset from: {dexycb_dir_path}")
    logging.info(f"Using URDFs from: {urdf_dir_path}")
    viz_retarget_only(
        dexycb_dir=dexycb_dir_path,
        urdf_dir=urdf_dir_path,
        robot_names=tuple(robots) if robots is not None else None,
        fps=fps,
        data_id=data_id,
        headless=headless,
    )


if __name__ == "__main__":
    tyro.cli(main)
