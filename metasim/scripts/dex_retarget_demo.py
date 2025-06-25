from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import tyro
from typing import Optional, List, Tuple
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from metasim.utils.dex_util.constants import HandType, RobotName
from metasim.utils.dex_util.dataset import DexYCBVideoDataset
from metasim.utils.dex_util.retargeting_config import RetargetingConfig
from metasim.utils.dex_util.hand_robot_viewer import RobotHandDatasetSAPIENViewer

# For numpy compatibility
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
    robot_names: Optional[Tuple[RobotName]],
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
    viewer = RobotHandDatasetSAPIENViewer(
        list(robot_names), HandType.right, headless=headless
    )
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps=fps)

def main(
    dexycb_dir: str,
    urdf_dir: str,
    robots: Optional[List[RobotName]] = None,
    fps: int = 10,
    data_id: int = 0,
    headless: bool = False,
):
    """
    Render human hand + robot hand trajectories with object, and export video if headless.
    """
    dexycb_dir = Path(dexycb_dir).absolute()
    urdf_dir = Path(urdf_dir).absolute()
    if not dexycb_dir.exists():
        raise FileNotFoundError(f"DexYCB dir not found: {dexycb_dir}")
    if not urdf_dir.exists():
        raise FileNotFoundError(f"URDF dir not found: {urdf_dir}")

    print(f"[INFO] Using DexYCB dataset from: {dexycb_dir}")
    print(f"[INFO] Using URDFs from: {urdf_dir}")
    viz_retarget_only(
        dexycb_dir=dexycb_dir,
        urdf_dir=urdf_dir,
        robot_names=robots,
        fps=fps,
        data_id=data_id,
        headless=headless,
    )

if __name__ == "__main__":
    tyro.cli(main)
