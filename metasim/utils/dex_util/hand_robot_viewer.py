from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sapien
from pytransform3d import rotations
from tqdm import trange

from metasim.utils.dex_util import yourdfpy as urdf
from metasim.utils.dex_util.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from metasim.utils.dex_util.hand_viewer import HandDatasetSAPIENViewer
from metasim.utils.dex_util.retargeting_config import RetargetingConfig
from metasim.utils.dex_util.seq_retarget import SeqRetargeting


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    """A viewer for rendering robot hand retargeting alongside human hand data from DexYCB dataset."""

    def __init__(
        self,
        robot_names: list[RobotName],
        hand_type: HandType,
        headless=False,
    ):
        super().__init__(headless=headless)

        self.robot_names = robot_names
        self.robots: list[Any] = []
        self.robot_file_names: list[str] = []
        self.retargetings: list[SeqRetargeting] = []
        self.retarget2sapien: list[np.ndarray] = []
        self.hand_type = hand_type

        # Load optimizer and filter
        if self.scene is not None:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True
            for robot_name in robot_names:
                config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)
                if config_path is None:
                    continue

                # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
                override = dict(add_dummy_free_joint=True)
                config = RetargetingConfig.load_from_file(config_path, override=override)
                retargeting = config.build()
                robot_file_name = Path(config.urdf_path).stem
                self.robot_file_names.append(robot_file_name)
                self.retargetings.append(retargeting)

                # Build robot
                urdf_path = Path(config.urdf_path)
                # if "glb" not in urdf_path.stem:
                #     urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
                robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
                urdf_name = urdf_path.name
                temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
                temp_path = f"{temp_dir}/{urdf_name}"
                robot_urdf.write_xml_file(temp_path)

                robot = loader.load(temp_path)
                self.robots.append(robot)
                sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
                retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
                self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: dict):
        """Load object and hand data from DexYCB dataset."""
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: dict, fps=5, y_offset=0.8):
        """Render the data from DexYCB dataset."""
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        # self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p([1.5, global_y_offset, 1])
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(hand_pose_start[0, 0:3])
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        if joint is not None:
            for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
                retargeting.warm_start(
                    joint[0, :],
                    wrist_quat,
                    hand_type=self.hand_type,
                    is_mano_convention=True,
                )

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))  # type: ignore
                if self.camera_pose is not None:
                    pose = self.camera_pose * pose
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(pose_offsets[copy_ind] * pose)

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
                if joint is not None:
                    indices = retargeting.optimizer.target_link_human_indices
                    ref_value = joint[indices, :]
                    qpos = retargeting.retarget(ref_value)[retarget2sapien]
                    robot.set_qpos(qpos)

            if self.scene is not None:
                self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()
