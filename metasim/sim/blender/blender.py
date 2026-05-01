from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import bpy
import imageio.v3 as iio
import numpy as np
import torch
from mathutils import Matrix, Quaternion, Vector

from metasim.queries.base import BaseQueryType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import (
    CameraState,
    CompatActionInput,
    DictEnvState,
    Obs,
    Reward,
    RobotState,
    Success,
    TensorState,
    Termination,
)


def import_mesh(path):
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif extension == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError("bad mesh extension")

    return bpy.context.object


_BLENDER_NUMERIC_SUFFIX_RE = re.compile(r"\.\d{3}$")


def _normalized_blender_name(name: str) -> str:
    """Return a Blender object name without Blender's duplicate numeric suffix."""
    return _BLENDER_NUMERIC_SUFFIX_RE.sub("", name)


def _choose_body_object(candidates: list) -> object:
    """Choose the transform object that should receive a MetaSim body pose."""
    if not candidates:
        raise ValueError("cannot choose from an empty body object candidate list")

    empty_with_children = [obj for obj in candidates if getattr(obj, "type", None) == "EMPTY" and len(obj.children) > 0]
    if empty_with_children:
        return sorted(empty_with_children, key=lambda obj: obj.name)[0]

    empties = [obj for obj in candidates if getattr(obj, "type", None) == "EMPTY"]
    if empties:
        return sorted(empties, key=lambda obj: obj.name)[0]

    return sorted(candidates, key=lambda obj: obj.name)[0]


def _matrix_from_root_state(root_state: torch.Tensor) -> Matrix:
    """Convert a MetaSim root/body state row to a Blender world matrix."""
    state = root_state.detach().cpu().flatten()
    position = Vector((float(state[0]), float(state[1]), float(state[2])))
    quat = Quaternion((float(state[3]), float(state[4]), float(state[5]), float(state[6])))
    return Matrix.Translation(position) @ quat.to_matrix().to_4x4()


def _apply_root_state_to_object(obj, root_state: torch.Tensor) -> None:
    """Apply translation and rotation while preserving the object's visual scale."""
    bpy.context.view_layer.update()
    matrix = _matrix_from_root_state(root_state)
    world_scale = obj.matrix_world.to_scale()
    desired_world = matrix @ Matrix.Diagonal((world_scale.x, world_scale.y, world_scale.z, 1.0))
    if obj.parent is None:
        obj.matrix_world = desired_world
    else:
        obj.matrix_basis = obj.parent.matrix_world.inverted() @ desired_world
    bpy.context.view_layer.update()


def _blender_body_name_alias(body_name: str) -> str:
    """Map known MetaSim robot body names to Blender/USD object names."""
    for prefix in ("left_", "right_"):
        hand_prefix = f"{prefix}hand_"
        if body_name.startswith(hand_prefix):
            return f"{prefix}{body_name[len(hand_prefix):]}"
    return body_name


class BlenderHandler(BaseSimHandler):
    set_states_refreshes = True

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self.context = bpy.context
        self._objs: dict[str, object] = {}
        self._body_objs: dict[str, dict[str, object]] = {}
        self._body_names: dict[str, list[str]] = {}
        self._camera_objs: dict[str, object] = {}
        self._last_camera_states: dict[str, CameraState] = {}
        self._last_tensor_state: TensorState | None = None
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="metasim_blender_"))

    def launch(self) -> None:
        super().launch()
        self._clear_scene()
        self._configure_render()
        self._add_lights()

        for obj_cfg in self.objects:
            if isinstance(obj_cfg, PrimitiveCubeCfg):
                self._objs[obj_cfg.name] = self._create_cube(obj_cfg)
            elif isinstance(obj_cfg, PrimitiveSphereCfg):
                self._objs[obj_cfg.name] = self._create_sphere(obj_cfg)
            elif isinstance(obj_cfg, PrimitiveCylinderCfg):
                self._objs[obj_cfg.name] = self._create_cylinder(obj_cfg)
            elif isinstance(obj_cfg, RigidObjCfg):
                self._objs[obj_cfg.name] = self._import_rigid_object(obj_cfg)
            elif isinstance(obj_cfg, ArticulationObjCfg):
                self._import_articulation(obj_cfg)
            else:
                raise ValueError(f"Unknown object type: {type(obj_cfg)}")

        for robot in self.robots:
            self._import_articulation(robot)

        self._add_cameras()

    def _clear_scene(self) -> None:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def _configure_render(self) -> None:
        scene = self.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(getattr(getattr(self.scenario, "render", None), "samples", 64) or 64)
        scene.cycles.use_denoising = True
        scene.view_settings.view_transform = "Standard"
        scene.render.film_transparent = False
        self._configure_cycles_device(str(getattr(getattr(self.scenario, "render", None), "device", "CPU") or "CPU"))

    def _configure_cycles_device(self, device: str) -> None:
        scene = self.context.scene
        normalized = device.upper()
        if normalized == "CPU":
            scene.cycles.device = "CPU"
            return

        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = normalized
        prefs.get_devices()
        enabled = 0
        for blender_device in prefs.devices:
            use_device = blender_device.type == normalized
            blender_device.use = use_device
            enabled += int(use_device)
        if enabled == 0:
            raise RuntimeError(f"Requested Blender/Cycles device {normalized!r} is not available")
        scene.cycles.device = "GPU"

    def _add_lights(self) -> None:
        bpy.ops.object.light_add(type="AREA", location=(0.0, -2.0, 3.0))
        light = bpy.context.object
        light.name = "metasim_key_area_light"
        light.data.energy = 500
        light.data.size = 4.0

    def _make_material(self, name: str, color: list[float] | None):
        mat = bpy.data.materials.new(name=f"{name}_material")
        mat.use_nodes = True
        rgba = tuple(float(c) for c in (color or [0.7, 0.7, 0.7])) + (1.0,)
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = rgba
        mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.55
        return mat

    def _create_cube(self, obj_cfg: PrimitiveCubeCfg):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=obj_cfg.default_position)
        obj = bpy.context.object
        obj.name = obj_cfg.name
        obj.scale = tuple(float(v) for v in obj_cfg.size)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(float(v) for v in obj_cfg.default_orientation)
        obj.data.materials.append(self._make_material(obj_cfg.name, obj_cfg.color))
        return obj

    def _create_sphere(self, obj_cfg: PrimitiveSphereCfg):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(obj_cfg.radius), location=obj_cfg.default_position)
        obj = bpy.context.object
        obj.name = obj_cfg.name
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(float(v) for v in obj_cfg.default_orientation)
        obj.data.materials.append(self._make_material(obj_cfg.name, obj_cfg.color))
        return obj

    def _create_cylinder(self, obj_cfg: PrimitiveCylinderCfg):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=float(obj_cfg.radius),
            depth=float(obj_cfg.height),
            location=obj_cfg.default_position,
        )
        obj = bpy.context.object
        obj.name = obj_cfg.name
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(float(v) for v in obj_cfg.default_orientation)
        obj.data.materials.append(self._make_material(obj_cfg.name, obj_cfg.color))
        return obj

    def _import_rigid_object(self, obj_cfg: RigidObjCfg):
        if obj_cfg.mesh_path is None:
            raise ValueError(f"Rigid object {obj_cfg.name!r} requires mesh_path for Blender")
        obj = import_mesh(obj_cfg.mesh_path)
        obj.name = obj_cfg.name
        return obj

    def _import_articulation(self, obj_cfg: ArticulationObjCfg) -> None:
        usd_path = obj_cfg.file_name("blender")
        if not usd_path:
            raise ValueError(f"{type(obj_cfg).__name__} {obj_cfg.name!r} requires usd_path for Blender")
        before_names = set(bpy.data.objects.keys())
        result = bpy.ops.wm.usd_import(filepath=str(usd_path))
        if "FINISHED" not in result:
            raise RuntimeError(f"Failed to import USD for {obj_cfg.name!r}: {result}")
        imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
        if not imported:
            raise RuntimeError(f"USD import for {obj_cfg.name!r} produced no Blender objects")
        self._register_body_objects(obj_cfg.name, imported)

    def _register_body_objects(self, obj_name: str, imported: list) -> None:
        grouped: dict[str, list] = {}
        for obj in imported:
            grouped.setdefault(_normalized_blender_name(obj.name), []).append(obj)
        selected = {name: _choose_body_object(candidates) for name, candidates in grouped.items()}
        self._body_objs[obj_name] = selected
        self._body_names[obj_name] = sorted(selected)

    def _add_cameras(self) -> None:
        for camera in self.cameras:
            if not isinstance(camera, PinholeCameraCfg):
                raise TypeError(f"Blender only supports PinholeCameraCfg, got {type(camera)!r}")
            cam_obj = self._create_camera(camera)
            self._camera_objs[camera.name] = cam_obj

    def _create_camera(self, camera: PinholeCameraCfg):
        bpy.ops.object.camera_add(location=camera.pos)
        obj = bpy.context.object
        obj.name = camera.name
        obj.data.name = f"{camera.name}_data"
        obj.data.lens = float(camera.focal_length)
        obj.data.sensor_width = float(camera.horizontal_aperture)
        obj.data.clip_start = float(camera.clipping_range[0])
        obj.data.clip_end = float(camera.clipping_range[1])
        direction = Vector(camera.look_at) - Vector(camera.pos)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj

    def _set_states(self, states: list[DictEnvState] | TensorState, env_ids: list[int] | None = None) -> None:
        if env_ids not in (None, [0]):
            raise ValueError("BlenderHandler currently supports only one environment")
        if isinstance(states, TensorState):
            self._apply_tensor_state(states)
            self._last_tensor_state = states
        elif isinstance(states, list):
            if len(states) != 1:
                raise ValueError("BlenderHandler currently supports a single DictEnvState")
            self._apply_dict_state(states[0])
            self._last_tensor_state = None
        else:
            raise TypeError(f"Unsupported Blender state type: {type(states)!r}")
        self.refresh_render()

    def _apply_tensor_state(self, state: TensorState) -> None:
        for obj_name, obj_state in state.objects.items():
            if obj_name in self._objs:
                _apply_root_state_to_object(self._objs[obj_name], obj_state.root_state[0])
        for robot_name, robot_state in state.robots.items():
            self._apply_robot_body_state(robot_name, robot_state)

    def _apply_robot_body_state(self, robot_name: str, robot_state: RobotState) -> None:
        body_map = self._body_objs.get(robot_name, {})
        missing = []
        for body_index, body_name in enumerate(robot_state.body_names):
            obj = body_map.get(body_name)
            if obj is None:
                obj = body_map.get(_blender_body_name_alias(body_name))
            if obj is None:
                missing.append(body_name)
                continue
            _apply_root_state_to_object(obj, robot_state.body_state[0, body_index])
        if missing:
            raise ValueError(f"Blender could not map bodies for {robot_name}: {missing[:10]}")

    def _apply_dict_state(self, state: DictEnvState) -> None:
        for obj_name, obj_state in state["objects"].items():
            if obj_name in self._objs:
                root = torch.zeros(13)
                root[:3] = obj_state["pos"]
                root[3:7] = obj_state["rot"]
                _apply_root_state_to_object(self._objs[obj_name], root)

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids not in (None, [0]):
            raise ValueError("BlenderHandler currently supports only one environment")
        if self._last_tensor_state is None:
            return TensorState(objects={}, robots={}, cameras=self._last_camera_states)
        return TensorState(
            objects=self._last_tensor_state.objects,
            robots=self._last_tensor_state.robots,
            cameras=self._last_camera_states,
            extras=self._last_tensor_state.extras,
        )

    def get_states(self, env_ids: list[int] | None = None, mode: str = "tensor") -> TensorState:
        if mode != "tensor":
            raise NotImplementedError("BlenderHandler only supports tensor state output")
        return super().get_states(env_ids=env_ids, mode="tensor")

    def reset(self, env_ids: list[int] | None = None) -> tuple[Obs, Any]:
        if env_ids is None:
            env_ids = [0]
        assert env_ids == [0]
        obs = self._get_observation()
        return obs, None

    def step(self, action: CompatActionInput) -> tuple[Obs, Reward, Success, Termination, Any]:
        _ = action
        raise NotImplementedError("BlenderHandler is render-only; step is not supported.")

    def render(self) -> None:
        self.refresh_render()

    def refresh_render(self) -> None:
        camera_states: dict[str, CameraState] = {}
        for camera in self.cameras:
            camera_states[camera.name] = self._render_camera(camera)
        self._last_camera_states = camera_states
        self._invalidate_state_caches()

    def _render_camera(self, camera: PinholeCameraCfg) -> CameraState:
        scene = self.context.scene
        scene.camera = self._camera_objs[camera.name]
        scene.render.resolution_x = int(camera.width)
        scene.render.resolution_y = int(camera.height)
        scene.render.resolution_percentage = 100
        image_path = self._tmp_dir / f"{camera.name}.png"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        rgb_np = np.asarray(iio.imread(image_path))[..., :3].copy()
        rgb = torch.from_numpy(rgb_np).to(torch.uint8).unsqueeze(0)
        return CameraState(
            rgb=rgb,
            depth=None,
            pos=torch.tensor([camera.pos], dtype=torch.float32),
            intrinsics=torch.tensor([camera.intrinsics], dtype=torch.float32),
        )

    def _simulate(self):
        self.refresh_render()

    def close(self) -> None:
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        _ = actions
        raise NotImplementedError("BlenderHandler is render-only; set_dof_targets is not supported.")

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        obj = self.object_dict.get(obj_name)
        joint_names = list(getattr(obj, "joint_names", []) or [])
        return sorted(joint_names) if sort else joint_names

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        body_names = list(self._body_names.get(obj_name, []))
        return sorted(body_names) if sort else body_names

    def _get_observation(self) -> Obs:
        if not self.cameras:
            return {}
        self.refresh_render()
        rgb = self._last_camera_states[self.cameras[0].name].rgb
        return {"rgb": rgb}

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


BlenderEnv = BlenderHandler
