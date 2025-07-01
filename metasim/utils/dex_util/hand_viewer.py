from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
from pytransform3d import transformations as pt
from sapien import internal_renderer as R
from tqdm import trange

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.sapien.sapien3 import Sapien3Handler
from metasim.utils.dex_util.dataset import YCB_CLASSES
from metasim.utils.dex_util.mano_layer import MANOLayer
from metasim.utils.setup_util import get_scene


def compute_smooth_shading_normal_np(vertices, indices):
    """Compute the vertex normal from vertices and triangles with numpy.

    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class HandDatasetSAPIENViewer:
    """A viewer for rendering hand data from DexYCB dataset using SAPIEN."""

    def __init__(self, headless=False, use_ray_tracing=False, scenario_cfg: ScenarioCfg | None = None):
        self.headless = headless

        # build ScenarioCfg
        scenario_cfg = ScenarioCfg(
            sim="sapien",
            headless=headless,
            try_add_table=True,
            robots=["InspireHandRight"],
        )
        if isinstance(scenario_cfg.scene, str):
            scenario_cfg.scene = get_scene(scenario_cfg.scene)

        self.scenario = scenario_cfg
        self.handler = Sapien3Handler(scenario_cfg)
        self.handler.launch()
        self.scene = self.handler.scene

        if not headless:
            self.viewer = self.handler.viewer
        else:
            self.camera = self.scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061]))

        # Renderer context
        self.internal_scene: R.Scene = self.scene.render_system._internal_scene
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        self.mano_layer: MANOLayer | None = None
        self.mano_face: np.ndarray | None = None
        self.camera_pose: sapien.Pose | None = None
        self.objects: list[sapien.Entity] = []
        self.nodes: list[R.Node] = []
        # self.internal_scene: R.Scene | None = None

    def clear_all(self):
        """Clear all objects and nodes from the scene."""
        for actor in self.objects:
            self.scene.remove_actor(actor)
        for _ in range(len(self.objects)):
            actor = self.objects.pop()
            self.scene.remove_actor(actor)
        self.clear_node()
        self.mano_layer = None

    def clear_node(self):
        """Clear all nodes from the internal scene."""
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)

    def load_object_hand(self, data: dict):
        """Load object and hand data from DexYCB dataset."""
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            self._load_ycb_object(ycb_id, ycb_mesh_file)

        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        pose_vec = pt.pq_from_transform(extrinsic_mat)
        self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

    def _load_ycb_object(self, ycb_id, ycb_mesh_file):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(ycb_mesh_file)
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])
        self.objects.append(actor)

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)
        return vertex, joint

    def _update_hand(self, vertex):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def render_dexycb_data(self, data: dict, fps=10):
        """Render hand data from DexYCB dataset."""
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        frame_num = hand_pose.shape[0]

        if self.headless:
            video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        step_per_frame = int(60 / fps)
        for i in trange(frame_num):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, _ = self._compute_hand_geometry(hand_pose_frame)
            if vertex is not None:
                self._update_hand(vertex)
            for k in range(len(self.objects)):
                pos_quat = object_pose_frame[k]
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.objects[k].set_pose(pose)
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
