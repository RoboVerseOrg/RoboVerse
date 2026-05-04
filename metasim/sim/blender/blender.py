from __future__ import annotations

import os
import tempfile

import bpy
import numpy as np
import torch
from loguru import logger as log
from mathutils import Matrix

from metasim.queries.base import BaseQueryType
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import Action, CameraState, Obs, Reward, Success, TensorState, Termination
from metasim.utils.camera_util import get_cam_params
from metasim.utils.math import matrix_from_quat

from .utils import delete_all
from .utils.camera_util import get_blender_camera_from_KRT
from .utils.render_util import (
    assign_pass_indices,
    configure_compositor,
    configure_cycles,
    enable_passes,
    find_rendered_file,
    read_exr_layer,
)
from .utils.robot_util import RobotVisualState, load_robot_visuals, update_robot_pose


def import_mesh(path: str) -> bpy.types.Object:
    """Import a mesh into the active scene and return the new object.

    Uses the modern ``bpy.ops.wm.*_import`` operators that exist on Blender 4.x and 5.x.
    Falls back to the legacy ``import_mesh.*`` ops on builds where the new ones are missing.
    """
    _, extension = os.path.splitext(path)
    extension = extension.lower()

    if extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif extension == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif extension == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif extension == ".fbx":
        if hasattr(bpy.ops.wm, "fbx_import"):
            bpy.ops.wm.fbx_import(filepath=path)
        else:
            bpy.ops.import_scene.fbx(filepath=path)
    elif extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".dae":
        if hasattr(bpy.ops.wm, "collada_import"):
            bpy.ops.wm.collada_import(filepath=path)
        else:
            # Blender 5.x's pip wheel ships no Collada importer — go via trimesh.
            from .utils.robot_util import import_mesh_as_glb

            obj = import_mesh_as_glb(path)
            if obj is None:
                raise RuntimeError(f"failed to import DAE: {path}")
            return obj
    else:
        raise ValueError(f"Unsupported mesh extension: {extension}")

    return bpy.context.object


def _to_rgb_tensor(rgba: np.ndarray) -> torch.Tensor:
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)


def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).unsqueeze(0)


def _to_int_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    rounded = np.rint(arr).astype(np.int32)
    return torch.from_numpy(np.ascontiguousarray(rounded)).unsqueeze(0)


class BlenderHandler(BaseSimHandler):
    """Render-only Blender backend.

    Exposes Blender as a ``BaseSimHandler`` that can pose meshes and produce
    photoreal RGB / depth / normal / instance-segmentation images. The handler
    does **not** simulate physics — drive ``set_states`` from another handler
    (e.g. via :class:`HybridHandler`) when you need dynamics.
    """

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self.context = bpy.context
        self._objs: dict[str, bpy.types.Object] = {}
        self._robots: dict[str, RobotVisualState] = {}
        self._cameras_blender: dict[str, bpy.types.Object] = {}
        self._instance_id2label: dict[int, str] = {}

    cycles_samples: int = 64
    """Cycles ``samples`` value applied during ``launch``."""
    cycles_adaptive_threshold: float = 0.01
    """Adaptive sampling noise threshold (lower = cleaner / slower)."""
    cycles_max_bounces: int = 8
    """Total path-trace bounces (boost for HDRI-heavy / specular scenes)."""

    def launch(self) -> None:
        super().launch()
        log.info(f"Launching Blender backend (bpy {bpy.app.version_string})")
        delete_all(self.context, ["MESH"])
        delete_all(self.context, ["CAMERA"])

        for obj_cfg in self.objects:
            if isinstance(obj_cfg, (PrimitiveCubeCfg, PrimitiveSphereCfg, PrimitiveCylinderCfg)):
                obj = self._spawn_primitive(obj_cfg)
                self._objs[obj_cfg.name] = obj
                continue
            if isinstance(obj_cfg, ArticulationObjCfg):
                self._load_articulated(obj_cfg)
                continue
            if isinstance(obj_cfg, RigidObjCfg):
                if obj_cfg.mesh_path:
                    obj = import_mesh(obj_cfg.mesh_path)
                    obj.name = obj_cfg.name
                    self._objs[obj_cfg.name] = obj
                elif getattr(obj_cfg, "urdf_path", None):
                    # Single-link rigid objects shipped as URDFs (libero objects):
                    # use the URDF loader so we get the mesh + per-region colors.
                    self._load_articulated(obj_cfg)
                else:
                    log.warning(f"Skipping {obj_cfg.name}: no mesh_path or urdf_path set")
            else:
                raise ValueError(f"Unknown object type: {type(obj_cfg)}")

        for robot_cfg in self.robots:
            self._load_articulated(robot_cfg)

        self._build_id2label()
        self._add_cameras()
        configure_cycles(
            self.context.scene,
            samples=self.cycles_samples,
            adaptive_threshold=self.cycles_adaptive_threshold,
            max_bounces=self.cycles_max_bounces,
        )

    def set_quality(
        self,
        samples: int,
        adaptive_threshold: float | None = None,
        max_bounces: int | None = None,
    ) -> None:
        """Tune Cycles quality on an already-launched scene."""
        self.cycles_samples = int(samples)
        if adaptive_threshold is not None:
            self.cycles_adaptive_threshold = float(adaptive_threshold)
        if max_bounces is not None:
            self.cycles_max_bounces = int(max_bounces)
        configure_cycles(
            self.context.scene,
            samples=self.cycles_samples,
            adaptive_threshold=self.cycles_adaptive_threshold,
            max_bounces=self.cycles_max_bounces,
        )

    def _spawn_primitive(self, cfg) -> bpy.types.Object:
        """Spawn a primitive cube/sphere/cylinder with the cfg's size + color."""
        if isinstance(cfg, PrimitiveCubeCfg):
            size = cfg.size
            # primitive_cube_add(size=1.0) gives side length 1; we then multiply
            # by ``cfg.size`` so the final side length matches the cfg (the older
            # ``size/2`` here was a bug — cubes came out half the intended size).
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
            obj = bpy.context.object
            obj.scale = (float(size[0]), float(size[1]), float(size[2]))
        elif isinstance(cfg, PrimitiveSphereCfg):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=float(cfg.radius), location=(0.0, 0.0, 0.0))
            obj = bpy.context.object
        elif isinstance(cfg, PrimitiveCylinderCfg):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=float(cfg.radius),
                depth=float(cfg.height),
                location=(0.0, 0.0, 0.0),
            )
            obj = bpy.context.object
        else:
            raise ValueError(f"Unknown primitive cfg type: {type(cfg).__name__}")

        obj.name = cfg.name
        # Bake scale into the mesh data so set_states' matrix_world doesn't re-scale.
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        color = list(cfg.color) if cfg.color is not None else [0.6, 0.6, 0.6]
        rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
        mat = bpy.data.materials.new(name=f"{cfg.name}_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.6
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        return obj

    def _load_articulated(self, cfg) -> None:
        """Load an articulated cfg (RobotCfg or ArticulationObjCfg) from URDF."""
        urdf_path = getattr(cfg, "urdf_path", None)
        if not urdf_path or not os.path.isfile(urdf_path):
            log.warning(f"Skipping {cfg.name}: no urdf_path or file not found ({urdf_path!r})")
            return
        pass_index = 1 + len(self._objs) + len(self._robots)
        self._robots[cfg.name] = load_robot_visuals(urdf_path, name_prefix=cfg.name, pass_index=pass_index)
        log.info(f"loaded {cfg.name}: {len(self._robots[cfg.name].visual_objs)} link visuals")

    def _build_id2label(self) -> None:
        """Assign pass indices to rigid objects (1..N) and robots (N+1..N+M)."""
        rigid_ids = assign_pass_indices(list(self._objs.values()))
        self._instance_id2label = dict(rigid_ids)
        for name, robot in self._robots.items():
            self._instance_id2label[robot.pass_index] = name

    def _set_states(self, states, env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert env_ids == [0], "BlenderHandler is single-env only"
        if isinstance(states, list):
            state0 = states[0]
        else:
            state0 = states
        # Two state shapes show up in roboverse:
        #   * flat:   {name: {pos, rot, dof_pos}}
        #   * nested: {"objects": {name: ...}, "robots": {name: ...}, ...}
        # The hybrid sim handler emits nested; the get_started smoke tests use flat.
        if "objects" in state0 or "robots" in state0:
            flat: dict = {}
            flat.update(state0.get("objects", {}) or {})
            flat.update(state0.get("robots", {}) or {})
            state0 = flat
        for obj_cfg in self.objects + self.robots:
            if obj_cfg.name not in state0:
                log.warning(f"Missing {obj_cfg.name} in states")
                continue
            entry = state0[obj_cfg.name]
            root_world = self._build_root_world(entry)

            if obj_cfg.name in self._robots:
                update_robot_pose(self._robots[obj_cfg.name], root_world, entry.get("dof_pos"))
                continue

            obj = self._objs.get(obj_cfg.name)
            if obj is None:
                continue
            obj.matrix_world = root_world

    @staticmethod
    def _build_root_world(entry) -> Matrix:
        pos = entry.get("pos")
        rot = entry.get("rot")
        pos_list = pos.tolist() if hasattr(pos, "tolist") else list(pos) if pos is not None else [0.0, 0.0, 0.0]
        if rot is not None:
            rot_mat = Matrix(matrix_from_quat(rot).tolist()).to_4x4()
        else:
            rot_mat = Matrix.Identity(4)
        return Matrix.Translation(pos_list) @ rot_mat

    def reset(self, env_ids: list[int] | None = None) -> tuple[Obs, dict | None]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert env_ids == [0]
        return self._get_states(env_ids=env_ids), None

    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, Termination, dict | None]:
        raise NotImplementedError("BlenderHandler is render-only; drive it via set_states from another handler.")

    def _set_dof_targets(self, actions: list[Action]) -> None:
        raise NotImplementedError("BlenderHandler is render-only; _set_dof_targets is not supported.")

    def _simulate(self) -> None:
        # Render-only; physics is delegated to a co-handler in hybrid setups.
        return

    def _add_cameras(self) -> None:
        for cam_cfg in self.cameras:
            self._build_camera(cam_cfg, cam_cfg.pos, cam_cfg.look_at)

    def _build_camera(self, cam_cfg, pos, look_at) -> bpy.types.Object:
        existing = self._cameras_blender.get(cam_cfg.name)
        if existing is not None:
            try:
                _ = existing.name  # touch RNA to detect stale references
                bpy.data.objects.remove(existing, do_unlink=True)
            except ReferenceError:
                pass  # already removed by a prior delete_all(["CAMERA"])

        kw = {"width": cam_cfg.width, "height": cam_cfg.height}
        for attr in ("focal_length", "horizontal_aperture"):
            if hasattr(cam_cfg, attr):
                kw[attr] = getattr(cam_cfg, attr)
        extrinsics, intrinsics = get_cam_params(
            cam_pos=torch.tensor(pos)[None, :],
            cam_look_at=torch.tensor(look_at)[None, :],
            **kw,
        )
        K = intrinsics.squeeze(0).numpy()
        R = extrinsics.squeeze(0)[:3, :3].numpy()
        T = extrinsics.squeeze(0)[:3, 3].numpy()
        cam_obj = get_blender_camera_from_KRT(K, R, T)
        cam_obj.name = cam_cfg.name
        self._cameras_blender[cam_cfg.name] = cam_obj
        return cam_obj

    def update_camera_pose(self, name: str, pos, look_at) -> None:
        """Re-place camera ``name`` at ``pos`` looking at ``look_at``.

        Useful for runtime augmentation. Recomputes K/R/T using the camera's
        existing intrinsics fields.
        """
        cam_cfg = next((c for c in self.cameras if c.name == name), None)
        if cam_cfg is None:
            raise KeyError(f"no camera named {name!r}")
        self._build_camera(cam_cfg, pos, look_at)
        self._invalidate_state_caches()

    def _render_camera(self, cam_cfg) -> CameraState:
        scene = self.context.scene
        scene.camera = self._cameras_blender[cam_cfg.name]
        scene.render.resolution_x = cam_cfg.width
        scene.render.resolution_y = cam_cfg.height

        wanted = set(cam_cfg.data_types)
        # 'instance_id_seg' (per-instance ids) is not separately implemented;
        # fall back to the object-index pass and mirror it into both fields.
        emit_instance_id_seg = "instance_id_seg" in wanted
        if emit_instance_id_seg:
            log.warning("BlenderHandler: 'instance_id_seg' falls back to 'instance_seg' (object-index pass)")
            wanted.add("instance_seg")

        enable_passes(scene, wanted)

        with tempfile.TemporaryDirectory(prefix="blender_render_") as tmpdir:
            prefixes = configure_compositor(scene, wanted, tmpdir)
            scene.render.filepath = os.path.join(tmpdir, "_main_")
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=False)

            rgb = depth = normal = inst_seg = None
            if "rgb" in prefixes:
                rgba = read_exr_layer(find_rendered_file(tmpdir, prefixes["rgb"]), "rgb")
                rgb = _to_rgb_tensor(rgba)
            if "depth" in prefixes:
                depth = _to_float_tensor(read_exr_layer(find_rendered_file(tmpdir, prefixes["depth"]), "depth"))
            if "normal" in prefixes:
                normal = _to_float_tensor(read_exr_layer(find_rendered_file(tmpdir, prefixes["normal"]), "normal"))
            if "instance_seg" in prefixes:
                inst_seg = _to_int_mask_tensor(
                    read_exr_layer(find_rendered_file(tmpdir, prefixes["instance_seg"]), "instance_seg")
                )

        id2label = dict(self._instance_id2label) if inst_seg is not None else None
        return CameraState(
            rgb=rgb,
            depth=depth,
            normal=normal,
            instance_seg=inst_seg,
            instance_seg_id2label=id2label,
            instance_id_seg=inst_seg if emit_instance_id_seg else None,
            instance_id_seg_id2label=id2label if emit_instance_id_seg else None,
        )

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is not None:
            assert env_ids == [0], "BlenderHandler is single-env only"
        # Make sure the dependency graph reflects the latest set_states() calls.
        self.context.view_layer.update()
        camera_states = {cam_cfg.name: self._render_camera(cam_cfg) for cam_cfg in self.cameras}
        return TensorState(objects={}, robots={}, cameras=camera_states, extras={})

    def render(self) -> None:
        self._get_states()

    def refresh_render(self) -> None:
        self.context.view_layer.update()

    def close(self) -> None:
        return
