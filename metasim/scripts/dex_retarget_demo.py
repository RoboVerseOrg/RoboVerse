from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from metasim.utils.dex_util.constants import HandType, RobotName
from metasim.utils.dex_util.dataset import DexYCBVideoDataset
from metasim.utils.dex_util.retargeting_config import RetargetingConfig


def setup_scene(headless: bool = False):
    """
    Create a SAPIEN scene with basic lighting and ground, and return the scene and viewer/camera.
    """
    # Scene
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)

    # Lighting
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_directional_light([1, -1, -1], [2, 2, 2], shadow=True)
    scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
    scene.set_ambient_light([0.2, 0.2, 0.2])

    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(1.5, 0, 1)
        viewer.set_camera_rpy(0, -0.8, 3.14)
        viewer.control_window.toggle_origin_frame(False)
        return scene, viewer, None
    else:
        camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
        camera.set_local_pose(sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061]))
        return scene, None, camera


def load_robot_and_retarget(scene, robot_name: RobotName, hand_type: HandType):
    """
    Load a single robot hand into the scene and build its retargeting pipeline.
    Returns: (sapien.Articulation, SeqRetargeting, retarget2sapien indices)
    """
    # Prepare loader
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    # Load retargeting config
    cfg_path = RetargetingConfig._DEFAULT_URDF_DIR  # ensure default dir set externally
    config_path = cfg_path / f"{robot_name.name}_{hand_type.name}.yml"
    override = {"add_dummy_free_joint": True}
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeter = config.build()

    # Load URDF into SAPIEN
    urdf_path = Path(config.urdf_path)
    if "glb" not in urdf_path.stem:
        urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
    art = loader.load(str(urdf_path))

    # Map retarget indices to SAPIEN joint order
    sap_joints = [j.name for j in art.get_active_joints()]
    ret2sap = np.array([retargeter.optimizer.target_joint_names.index(n) for n in sap_joints], dtype=int)

    return art, retargeter, ret2sap


def main(
    dexycb_dir: str,
    robots: list[RobotName] | None = None,
    fps: int = 10,
    headless: bool = False,
):
    """
    Visualize retargeted robot hands only, without rendering any objects.

    Args:
        dexycb_dir: root path to DexYCB dataset
        robots: list of RobotName enums to visualize
        fps: frames per second to render
        headless: whether to run in headless mode
    """
    data_root = Path(dexycb_dir)
    if not data_root.exists():
        raise ValueError(f"DexYCB dir {data_root} does not exist.")

    # Prepare dataset
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    data_id = 4  # choose data index
    sample = dataset[data_id]
    hand_pose = np.array(sample["hand_pose"])

    # Setup scene
    scene, viewer, camera = setup_scene(headless)

    # Set URDF dir for config
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    # Load robots and retargeters
    if robots is None:
        raise ValueError("Please specify at least one robot to visualize.")
    loaded = [load_robot_and_retarget(scene, r, HandType.right) for r in robots]

    # Warm start each retargeter
    wrist_quat = hand_pose[0, 0:4]
    wrist_pos = hand_pose[0, 4:7]
    for art, retargeter, _ in loaded:
        retargeter.warm_start(wrist_pos, wrist_quat, hand_type=HandType.right, is_mano_convention=True)

    # Render loop
    scene.update_render()
    step = int(60 / fps)
    for i in range(hand_pose.shape[0]):
        hp = hand_pose[i]
        values = hp.reshape(-1, 51)[:, :48]
        ref = values  # sequence of 48-dim poses? adjust if necessary
        # For each robot, retarget and apply
        for art, retargeter, ret2sap in loaded:
            q = retargeter.retarget(ref)
            q_sap = np.zeros(art.dof)
            q_sap[ret2sap] = q
            art.set_qpos(q_sap.tolist())

        # Render frame
        scene.update_render()
        if viewer:
            for _ in range(step):
                viewer.render()
        elif camera:
            camera.take_picture()

    # Pause at end
    if viewer:
        viewer.paused = True
        viewer.render()


if __name__ == "__main__":
    tyro.cli(main)
