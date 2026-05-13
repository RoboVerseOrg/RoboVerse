from __future__ import annotations

import ctypes
import os
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import cos, sin, tau
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
from metasim.utils.state import state_tensor_to_nested

from .importers import MJCF_SUFFIXES, USD_SUFFIXES, import_usd_visuals
from .lights import add_scenario_lights


def import_mesh(path):
    before_names = set(bpy.data.objects.keys())
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    # Blender 4.x → 5.x renamed mesh importers: ``bpy.ops.import_mesh.ply``
    # → ``bpy.ops.wm.ply_import``, same for STL. Try the new API first, fall
    # back to the legacy one for older Blender installs.
    if extension == ".ply":
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
        bpy.ops.import_scene.fbx(filepath=path)
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif extension == ".msh":
        return _import_mujoco_msh(path)
    elif extension == ".dae":
        # COLLADA is supported by every Blender version via wm.collada_import.
        bpy.ops.wm.collada_import(filepath=path)
    else:
        raise ValueError(f"bad mesh extension: {extension!r}")

    imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
    active = bpy.context.object
    if active in imported:
        return active
    mesh_objects = [obj for obj in imported if getattr(obj, "type", None) == "MESH"]
    if mesh_objects:
        return mesh_objects[0]
    return active


def _import_mujoco_msh(path: str):
    """Import a MuJoCo binary mesh (``.msh``) into the active scene.

    Format (little-endian):
        uint32 nvertex, uint32 nnormal, uint32 ntexcoord, uint32 nface
        float32 * 3 * nvertex   (vertex positions)
        float32 * 3 * nnormal   (per-vertex normals)
        float32 * 2 * ntexcoord (per-vertex texcoords)
        int32   * 3 * nface     (triangle indices)

    Used widely by LIBERO / robosuite / Open6DOR-derived assets. Blender has
    no built-in .msh importer, so build the mesh directly with from_pydata.
    """
    with open(path, "rb") as fh:
        header = fh.read(16)
        if len(header) < 16:
            raise ValueError(f"MuJoCo .msh file {path!r} is shorter than 16-byte header")
        nv, nn, nt, nf = struct.unpack("<IIII", header)
        vertex_bytes = fh.read(nv * 3 * 4)
        # Normals are recomputed by Blender from polygons, skip.
        fh.seek(nn * 3 * 4, os.SEEK_CUR)
        texcoord_bytes = fh.read(nt * 2 * 4)
        face_bytes = fh.read(nf * 3 * 4)

    vertices = np.frombuffer(vertex_bytes, dtype="<f4").reshape((nv, 3)) if nv else np.empty((0, 3))
    faces = np.frombuffer(face_bytes, dtype="<i4").reshape((nf, 3)) if nf else np.empty((0, 3), dtype=np.int32)
    texcoords = (
        np.frombuffer(texcoord_bytes, dtype="<f4").reshape((nt, 2)) if nt else None
    )

    name = Path(path).stem
    mesh_data = bpy.data.meshes.new(f"{name}_mesh")
    mesh_data.from_pydata([tuple(v) for v in vertices.tolist()], [], [tuple(f) for f in faces.tolist()])
    mesh_data.update()

    if texcoords is not None and len(texcoords) == nv and nf > 0:
        # MuJoCo .msh stores per-vertex UVs; Blender wants per-loop UVs (one
        # per triangle corner). Index into the per-vertex array using each
        # polygon's loop's vertex_index. Flip V because mujoco's convention
        # is top-left origin and Blender's is bottom-left.
        uv_layer = mesh_data.uv_layers.new(name="UVMap")
        loop_uvs = np.empty((len(mesh_data.loops), 2), dtype=np.float32)
        for loop in mesh_data.loops:
            u, v = texcoords[loop.vertex_index]
            loop_uvs[loop.index] = (float(u), 1.0 - float(v))
        uv_layer.data.foreach_set("uv", loop_uvs.reshape(-1))

    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj


_BLENDER_NUMERIC_SUFFIX_RE = re.compile(r"\.\d{3}$")
_MATERIAL_HEX_COLOR_RE = re.compile(r"([0-9A-Fa-f]{6})(?:\.\d{3})?$")
_TRANSMISSIVE_ALPHA_EPSILON = 1e-6
_CYCLES_GPU_DEVICE_PREFERENCE = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")
_BLENDER_RGB_READBACK_FORMAT = "BMP"
_BLENDER_RGB_READBACK_SUFFIX = "bmp"
_CYCLES_FINAL_RENDER_SETTINGS = {
    "samples": 128,
    "denoising_prefilter": "ACCURATE",
    "denoising_quality": "HIGH",
    "max_bounces": 10,
    "diffuse_bounces": 4,
    "glossy_bounces": 4,
    "transmission_bounces": 10,
    "transparent_max_bounces": 10,
    "caustics_reflective": True,
    "caustics_refractive": True,
}


@dataclass(frozen=True)
class UsdMaterialSpec:
    """Authored USD material inputs resolved from the source stage."""

    name: str
    base_color: tuple[float, float, float, float] | None = None
    base_color_texture: Path | None = None
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None


class UsdPythonBindingsUnavailable(RuntimeError):
    """Raised when optional pxr/USD Python bindings cannot be imported."""


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _scene_orientation_wxyz(scene_cfg) -> tuple[float, float, float, float]:
    """Return a Blender wxyz quaternion for a SceneCfg.

    IsaacSim applies ``SceneCfg.quat`` directly as ``(w, x, y, z)``.
    Blender must preserve that convention to render the same room pose.
    """
    default_orientation = getattr(scene_cfg, "default_orientation", None)
    if default_orientation is not None:
        values = tuple(float(value) for value in default_orientation)
        if len(values) != 4:
            raise ValueError(
                f"Scene {getattr(scene_cfg, 'name', None)!r} orientation must have 4 values, got {values!r}"
            )
        return values

    quat = getattr(scene_cfg, "quat", None)
    if quat is None:
        return (1.0, 0.0, 0.0, 0.0)
    values = tuple(float(value) for value in quat)
    if len(values) != 4:
        raise ValueError(f"Scene {getattr(scene_cfg, 'name', None)!r} orientation must have 4 values, got {values!r}")
    return values


def _scene_usd_xform_matrix(scene_cfg) -> Matrix:
    position = getattr(scene_cfg, "default_position", None) or (0.0, 0.0, 0.0)
    scale = getattr(scene_cfg, "scale", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
    orientation = Quaternion(_scene_orientation_wxyz(scene_cfg))
    return (
        orientation.to_matrix().to_4x4()
        @ Matrix.Translation(Vector(tuple(float(value) for value in position[:3])))
        @ Matrix.Diagonal(tuple(float(value) for value in scale[:3]) + (1.0,))
    )


def _srgb_channel_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _rgba_srgb_to_linear(rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        _srgb_channel_to_linear(rgba[0]),
        _srgb_channel_to_linear(rgba[1]),
        _srgb_channel_to_linear(rgba[2]),
        rgba[3],
    )


def _normalized_blender_name(name: str) -> str:
    """Return a Blender object name without Blender's duplicate numeric suffix."""
    return _BLENDER_NUMERIC_SUFFIX_RE.sub("", name)


def _normalized_material_name(name: str) -> str:
    """Return a material key shared by Blender names and asset-file names."""
    return _normalized_blender_name(name).replace(".", "_")


def _material_name_candidates(name: str) -> tuple[str, ...]:
    normalized = _normalized_material_name(name)
    candidates = [normalized]
    if normalized.endswith("-material"):
        candidates.append(normalized.removesuffix("-material"))
    for prefix in ("material_", "material-"):
        for candidate in tuple(candidates):
            if candidate.startswith(prefix) and len(candidate) > len(prefix):
                candidates.append(candidate[len(prefix) :])
    return tuple(dict.fromkeys(candidates))


def _is_visible_object(obj) -> bool:
    try:
        return bool(obj.visible_get())
    except RuntimeError:
        return False


def _has_collection_instance_child(obj) -> bool:
    return any(
        getattr(child, "instance_type", None) == "COLLECTION" and getattr(child, "instance_collection", None) is not None
        for child in getattr(obj, "children", ())
    )


def _has_renderable_descendant(obj) -> bool:
    if getattr(obj, "type", None) == "MESH":
        return True
    if _has_collection_instance_child(obj):
        return True
    return any(_has_renderable_descendant(child) for child in getattr(obj, "children", ()))


def _choose_body_object(candidates: list) -> object:
    """Choose the transform object that should receive a MetaSim body pose."""
    if not candidates:
        raise ValueError("cannot choose from an empty body object candidate list")

    candidate_groups = [
        [
            obj
            for obj in candidates
            if _is_visible_object(obj)
            and getattr(obj, "type", None) == "EMPTY"
            and _has_renderable_descendant(obj)
        ],
        [obj for obj in candidates if getattr(obj, "type", None) == "EMPTY" and _has_renderable_descendant(obj)],
        [obj for obj in candidates if getattr(obj, "type", None) == "EMPTY" and len(obj.children) > 0],
        [obj for obj in candidates if _is_visible_object(obj) and getattr(obj, "type", None) == "EMPTY"],
        [obj for obj in candidates if getattr(obj, "type", None) == "EMPTY"],
        [obj for obj in candidates if _is_visible_object(obj)],
        candidates,
    ]
    for group in candidate_groups:
        if group:
            return sorted(group, key=lambda obj: obj.name)[0]

    raise RuntimeError("unreachable")


def _matrix_from_root_state(root_state: torch.Tensor) -> Matrix:
    """Convert a MetaSim root/body state row to a Blender world matrix."""
    state = root_state.detach().cpu().flatten()
    position = Vector((float(state[0]), float(state[1]), float(state[2])))
    quat = Quaternion((float(state[3]), float(state[4]), float(state[5]), float(state[6])))
    return Matrix.Translation(position) @ quat.to_matrix().to_4x4()


def _apply_root_state_to_object(obj, root_state: torch.Tensor, scale_override: Vector | None = None) -> None:
    """Apply translation and rotation while preserving the object's visual scale."""
    bpy.context.view_layer.update()
    matrix = _matrix_from_root_state(root_state)
    world_scale = scale_override if scale_override is not None else obj.matrix_world.to_scale()
    desired_world = matrix @ Matrix.Diagonal((world_scale.x, world_scale.y, world_scale.z, 1.0))
    if obj.parent is None:
        obj.matrix_world = desired_world
    else:
        obj.matrix_basis = obj.parent.matrix_world.inverted() @ desired_world
    bpy.context.view_layer.update()


def _robot_body_pose_scale(obj) -> Vector:
    """Return the world scale to use when replaying a simulator body pose."""
    bpy.context.view_layer.update()
    world_scale = obj.matrix_world.to_scale()
    if getattr(obj, "type", None) == "EMPTY" and _has_collection_instance_child(obj):
        max_scale = max(abs(world_scale.x), abs(world_scale.y), abs(world_scale.z))
        min_scale = min(abs(world_scale.x), abs(world_scale.y), abs(world_scale.z))
        if max_scale <= 0.1 and abs(max_scale - min_scale) <= 1e-6:
            return Vector((1.0, 1.0, 1.0))
    return world_scale


def _resolve_robot_body_object(body_map: dict[str, object], body_name: str) -> object | None:
    alias = _blender_body_name_alias(body_name)
    exact_candidates = [body_map.get(name) for name in (body_name, alias)]
    visual_candidates = [body_map.get(f"{name}_visual") for name in (alias, body_name)]
    for obj in exact_candidates:
        if obj is not None and _has_renderable_descendant(obj):
            return obj
    for obj in visual_candidates:
        if obj is not None and _has_renderable_descendant(obj):
            return obj
    for obj in exact_candidates:
        if obj is not None:
            return obj
    return next((obj for obj in visual_candidates if obj is not None), None)


def _object_hierarchy_depth(obj) -> int:
    depth = 0
    parent = getattr(obj, "parent", None)
    while parent is not None:
        depth += 1
        parent = getattr(parent, "parent", None)
    return depth


def _rgba_from_material(material) -> tuple[float, float, float, float]:
    match = _MATERIAL_HEX_COLOR_RE.search(material.name)
    if match:
        color = match.group(1)
        rgba = tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4)) + (1.0,)
        return _rgba_srgb_to_linear(rgba)
    diffuse = tuple(float(value) for value in getattr(material, "diffuse_color", (0.7, 0.7, 0.7, 1.0)))
    if len(diffuse) >= 4:
        return _rgba_srgb_to_linear(diffuse[:4])
    if len(diffuse) >= 3:
        return _rgba_srgb_to_linear(diffuse[:3] + (1.0,))
    return _rgba_srgb_to_linear((0.7, 0.7, 0.7, 1.0))


def _rgba_from_text(text: str) -> tuple[float, float, float, float] | None:
    values = [float(value) for value in text.split()]
    if len(values) < 3:
        return None
    rgba = tuple(values[:4]) if len(values) >= 4 else tuple(values[:3]) + (1.0,)
    return _rgba_srgb_to_linear(rgba)


def _resolve_asset_path(path: str | Path, base_dir: Path | None = None) -> Path | None:
    raw_path = str(path)
    if raw_path.startswith("file://"):
        raw_path = raw_path[len("file://") :]
    elif raw_path.startswith("package://"):
        raw_path = raw_path[len("package://") :]

    candidate = Path(raw_path)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    if base_dir is not None and not candidate.is_absolute():
        candidates.append(base_dir / candidate)
    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)

    for resolved in candidates:
        if resolved.exists():
            return resolved.resolve()
    return None


def _preload_current_libpython() -> None:
    """Make the current CPython shared library visible before loading Kit USD bindings."""
    lib_name = f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
    candidate = Path(sys.prefix) / "lib" / lib_name
    if candidate.is_file():
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


def _usd_python_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env_python_path = os.environ.get("METASIM_USD_PYTHON_PATH")
    if env_python_path:
        candidates.append(Path(env_python_path))
    env_root = os.environ.get("METASIM_USD_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "lib" / "python")

    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    packman_root = Path.home() / ".cache" / "packman" / "chk"
    if packman_root.is_dir():
        candidates.extend(sorted(packman_root.glob(f"usd.{py_tag}*/**/lib/python"), reverse=True))

    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def _import_pxr_modules():
    try:
        from pxr import Usd, UsdShade

        return Usd, UsdShade
    except Exception as direct_err:
        last_err: Exception = direct_err

    try:
        _preload_current_libpython()
    except Exception as preload_err:
        last_err = preload_err

    for python_path in _usd_python_candidate_paths():
        path_str = str(python_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            from pxr import Usd, UsdShade

            return Usd, UsdShade
        except Exception as err:
            last_err = err

    raise UsdPythonBindingsUnavailable(
        "USD Python bindings are required to import authored USD materials for Blender. "
        "Install pxr bindings, or set METASIM_USD_PYTHON_PATH / METASIM_USD_ROOT to a Kit USD Python path."
    ) from last_err


def _usd_value_as_color(value) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = tuple(float(item) for item in value)
    except TypeError:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        return (_clamp01(scalar), _clamp01(scalar), _clamp01(scalar), 1.0)
    if len(values) >= 4:
        return tuple(_clamp01(item) for item in values[:4])
    if len(values) >= 3:
        return tuple(_clamp01(item) for item in values[:3]) + (1.0,)
    if len(values) == 1:
        return (_clamp01(values[0]), _clamp01(values[0]), _clamp01(values[0]), 1.0)
    return None


def _usd_value_as_scalar(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        values = tuple(float(item) for item in value)
    except TypeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if not values:
        return None
    return sum(values[: min(3, len(values))]) / min(3, len(values))


def _usd_embodiedgen_texture_fallback_path(raw: str, base_dir: Path) -> Path | None:
    candidate = Path(raw)
    if candidate.is_absolute() or not raw.startswith("textures/"):
        return None

    basename = candidate.name
    if not basename:
        return None

    for fallback in (
        base_dir / ".." / "result" / "mesh" / basename,
        base_dir / ".." / "mjcf" / "mesh" / basename,
        base_dir / ".." / basename,
    ):
        if fallback.is_file():
            return fallback.resolve()
    return None


def _usd_asset_path(value, base_dir: Path) -> Path | None:
    if value is None:
        return None
    raw = str(getattr(value, "resolvedPath", "") or getattr(value, "path", "") or value).strip()
    if not raw:
        return None
    raw = raw.strip("@")
    if raw in {"DefaultTexture", "./DefaultTexture"}:
        return None
    resolved = _resolve_asset_path(raw, base_dir)
    if resolved is not None:
        return resolved if resolved.is_file() else None
    return _usd_embodiedgen_texture_fallback_path(raw, base_dir)


def _usd_input_value(values: dict[str, object], names: tuple[str, ...]):
    for name in names:
        if name in values:
            return values[name]
    return None


def _usd_shader_input_values(material_prim, UsdShade) -> dict[str, object]:
    values: dict[str, object] = {}
    for child in material_prim.GetChildren():
        if child.GetTypeName() != "Shader":
            continue
        child_name = child.GetName().lower()
        shader = UsdShade.Shader(child)
        child_values: dict[str, object] = {}
        for shader_input in shader.GetInputs():
            child_values[shader_input.GetBaseName()] = shader_input.Get()
            values[shader_input.GetBaseName()] = child_values[shader_input.GetBaseName()]
        if any(token in child_name for token in ("diffuse", "basecolor", "albedo")):
            file_value = child_values.get("file")
            if file_value is not None:
                values["BaseColor_Tex"] = file_value
                values["IsBaseColorTex"] = 1.0
    return values


def _usd_material_specs_from_stage(usd_path: str | Path) -> dict[str, UsdMaterialSpec]:
    path = Path(usd_path)
    if not path.is_absolute():
        path = path.resolve()
    Usd, UsdShade = _import_pxr_modules()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage for Blender material import: {path}")

    specs: dict[str, UsdMaterialSpec] = {}
    for prim in stage.Traverse(Usd.PrimAllPrimsPredicate):
        if prim.GetTypeName() != "Material":
            continue
        material_name = prim.GetName()
        values = _usd_shader_input_values(prim, UsdShade)

        base_color = _usd_value_as_color(
            _usd_input_value(
                values,
                (
                    "BaseColor_Color",
                    "diffuse_color_constant",
                    "diffuseColor",
                    "base_color",
                    "diffuse_color",
                ),
            )
        )
        base_texture_value = _usd_input_value(values, ("BaseColor_Tex", "diffuse_texture", "albedo_texture"))
        texture_enabled = _usd_value_as_scalar(_usd_input_value(values, ("IsBaseColorTex", "use_diffuse_texture")))
        base_texture = _usd_asset_path(base_texture_value, path.parent) if (texture_enabled or 0.0) > 0.5 else None
        if base_texture is None and "BaseColor_Tex" not in values:
            base_texture = _usd_asset_path(base_texture_value, path.parent)

        roughness = _usd_value_as_scalar(
            _usd_input_value(values, ("reflection_roughness_constant", "roughness", "Roughness"))
        )
        if roughness is None:
            gloss = _usd_value_as_scalar(_usd_input_value(values, ("Gloss_Color", "glossiness", "Glossiness")))
            if gloss is not None:
                roughness = 1.0 - gloss
        metallic = _usd_value_as_scalar(
            _usd_input_value(values, ("Metallic_Color", "metallic", "metallic_constant", "Metallic"))
        )
        opacity = _usd_value_as_scalar(_usd_input_value(values, ("Opacity", "opacity", "opacity_constant")))

        if all(value is None for value in (base_color, base_texture, roughness, metallic, opacity)):
            continue
        spec = UsdMaterialSpec(
            name=material_name,
            base_color=base_color,
            base_color_texture=base_texture,
            metallic=_clamp01(metallic) if metallic is not None else None,
            roughness=_clamp01(roughness) if roughness is not None else None,
            opacity=_clamp01(opacity) if opacity is not None else None,
        )
        specs[material_name] = spec
        specs[_normalized_material_name(material_name)] = spec
    return specs


def _gf_matrix_to_blender(matrix) -> Matrix:
    """Convert a USD/Gf row-vector matrix into Blender's column-vector Matrix."""
    return Matrix(tuple(tuple(float(matrix[col][row]) for col in range(4)) for row in range(4)))


def _usd_prim_has_authored_references(prim) -> bool:
    has_authored_references = getattr(prim, "HasAuthoredReferences", None)
    if has_authored_references is not None:
        return bool(has_authored_references())
    return prim.GetMetadata("references") is not None


def _usd_reference_xforms_by_name(usd_path: str | Path) -> dict[str, list[Matrix]]:
    path = Path(usd_path)
    if not path.is_absolute():
        path = path.resolve()
    Usd, _UsdShade = _import_pxr_modules()
    from pxr import UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage for Blender reference transform import: {path}")

    entries: dict[str, list[tuple[str, Matrix]]] = {}
    for prim in stage.Traverse(Usd.PrimAllPrimsPredicate):
        if not _usd_prim_has_authored_references(prim):
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        entries.setdefault(prim.GetName(), []).append((str(prim.GetPath()), _gf_matrix_to_blender(world_matrix)))

    return {
        name: [matrix for _path, matrix in sorted(values, key=lambda item: item[0])]
        for name, values in entries.items()
    }


def _apply_usd_reference_xforms(usd_path: str | Path, imported: list) -> None:
    reference_xforms = _usd_reference_xforms_by_name(usd_path)
    if not reference_xforms:
        return
    imported_set = set(imported)
    root_objects = sorted(
        [obj for obj in imported if getattr(obj, "parent", None) not in imported_set],
        key=lambda obj: obj.name,
    )
    xform_indices = {name: 0 for name in reference_xforms}

    for obj in root_objects:
        name = _normalized_blender_name(obj.name)
        xforms = reference_xforms.get(name)
        if not xforms:
            continue
        index = min(xform_indices[name], len(xforms) - 1)
        xform_indices[name] += 1
        obj.matrix_basis = xforms[index] @ obj.matrix_basis


def _apply_usd_material_spec_to_material(material, spec: UsdMaterialSpec) -> None:
    base_color = spec.base_color or tuple(float(value) for value in material.diffuse_color[:4])
    opacity = spec.opacity if spec.opacity is not None else base_color[3]
    rgba = (base_color[0], base_color[1], base_color[2], opacity)

    material.use_nodes = True
    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    _set_principled_input(bsdf, "Base Color", rgba)
    _set_principled_input(bsdf, "Metallic", spec.metallic if spec.metallic is not None else 0.0)
    _set_principled_input(bsdf, "Roughness", spec.roughness if spec.roughness is not None else 0.55)
    _set_principled_input(bsdf, "Alpha", opacity)

    if spec.base_color_texture is not None:
        image = bpy.data.images.load(str(spec.base_color_texture), check_existing=True)
        try:
            image.colorspace_settings.name = "sRGB"
        except TypeError:
            pass
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = image
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    material.diffuse_color = rgba
    material.blend_method = "BLEND" if opacity < 1.0 - _TRANSMISSIVE_ALPHA_EPSILON else "OPAQUE"
    if hasattr(material, "use_screen_refraction"):
        material.use_screen_refraction = opacity < 1.0 - _TRANSMISSIVE_ALPHA_EPSILON
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False


def _apply_usd_material_specs(imported: list, specs: dict[str, UsdMaterialSpec]) -> None:
    seen_objects: set[str] = set()
    seen_collections: set[str] = set()
    seen_materials: set[str] = set()

    def visit_object(obj) -> None:
        if obj.name in seen_objects:
            return
        seen_objects.add(obj.name)
        if getattr(obj, "type", None) == "MESH":
            for slot in obj.material_slots:
                material = slot.material
                if material is None or material.name in seen_materials:
                    continue
                seen_materials.add(material.name)
                spec = next(
                    (specs[name] for name in _material_name_candidates(material.name) if name in specs),
                    None,
                )
                if spec is not None:
                    _apply_usd_material_spec_to_material(material, spec)
        collection = getattr(obj, "instance_collection", None)
        if collection is not None and collection.name not in seen_collections:
            seen_collections.add(collection.name)
            for child in collection.objects:
                visit_object(child)
        for child in getattr(obj, "children", ()):
            visit_object(child)

    for obj in imported:
        visit_object(obj)


def _apply_optional_usd_stage_enhancements(usd_path: str | Path, imported: list) -> None:
    """Apply pxr-backed USD reference transforms and materials when available."""
    try:
        _apply_usd_reference_xforms(usd_path, imported)
    except UsdPythonBindingsUnavailable:
        pass

    try:
        specs = _usd_material_specs_from_stage(usd_path)
    except UsdPythonBindingsUnavailable:
        return
    _apply_usd_material_specs(imported, specs)


def _material_rgba_from_collada(collada_path: str | Path) -> dict[str, tuple[float, float, float, float]]:
    path = Path(collada_path)
    root = ET.parse(path).getroot()
    effect_colors: dict[str, tuple[float, float, float, float]] = {}
    material_colors: dict[str, tuple[float, float, float, float]] = {}

    for effect in root.findall(".//{*}library_effects/{*}effect"):
        effect_id = effect.get("id")
        color_element = effect.find(".//{*}diffuse/{*}color")
        if effect_id is None or color_element is None or color_element.text is None:
            continue
        rgba = _rgba_from_text(color_element.text)
        if rgba is not None:
            effect_colors[effect_id] = rgba

    for material in root.findall(".//{*}library_materials/{*}material"):
        instance_effect = material.find("{*}instance_effect")
        if instance_effect is None:
            continue
        effect_id = (instance_effect.get("url") or "").lstrip("#")
        rgba = effect_colors.get(effect_id)
        if rgba is None:
            continue
        for name in (material.get("name"), material.get("id")):
            if name:
                material_name = _normalized_material_name(name.removesuffix("-material"))
                if material_name:
                    material_colors[material_name] = rgba

    return material_colors


def _material_rgba_from_urdf(urdf_path: str | Path) -> dict[str, tuple[float, float, float, float]]:
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    material_colors: dict[str, tuple[float, float, float, float]] = {}

    for material in root.findall(".//{*}material"):
        name = material.get("name")
        color = material.find("{*}color")
        if name is None or color is None:
            continue
        rgba_text = color.get("rgba")
        if rgba_text is None:
            continue
        rgba = _rgba_from_text(rgba_text)
        if rgba is not None:
            material_colors[_normalized_material_name(name)] = rgba

    for mesh in root.findall(".//{*}visual/{*}geometry/{*}mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_path = _resolve_asset_path(filename, path.parent)
        if mesh_path is not None and mesh_path.suffix.lower() == ".dae":
            material_colors.update(_material_rgba_from_collada(mesh_path))

    return material_colors


def _material_rgba_from_mjcf(mjcf_path: str | Path) -> dict[str, tuple[float, float, float, float]]:
    path = Path(mjcf_path)
    root = ET.parse(path).getroot()
    material_colors: dict[str, tuple[float, float, float, float]] = {}

    for material in root.findall(".//material"):
        name = material.get("name")
        rgba_text = material.get("rgba")
        if name is None or rgba_text is None:
            continue
        rgba = _rgba_from_text(rgba_text)
        if rgba is not None:
            material_colors[_normalized_material_name(name)] = rgba

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        mesh_path = _resolve_asset_path(filename, path.parent)
        if mesh_path is not None and mesh_path.suffix.lower() == ".dae":
            material_colors.update(_material_rgba_from_collada(mesh_path))

    return material_colors


def _asset_material_rgba(obj_cfg: ArticulationObjCfg | RigidObjCfg) -> dict[str, tuple[float, float, float, float]]:
    material_colors: dict[str, tuple[float, float, float, float]] = {}
    for asset_path, parser in (
        (getattr(obj_cfg, "mjcf_path", None), _material_rgba_from_mjcf),
        (getattr(obj_cfg, "urdf_path", None), _material_rgba_from_urdf),
    ):
        if not asset_path:
            continue
        resolved = _resolve_asset_path(asset_path)
        if resolved is not None:
            material_colors.update(parser(resolved))

    for asset_path in getattr(obj_cfg, "extra_resources", ()):
        resolved = _resolve_asset_path(asset_path)
        if resolved is None:
            continue
        if resolved.suffix.lower() == ".dae":
            material_colors.update(_material_rgba_from_collada(resolved))
        elif resolved.suffix.lower() == ".urdf":
            material_colors.update(_material_rgba_from_urdf(resolved))
        elif resolved.suffix.lower() in {".xml", ".mjcf"}:
            material_colors.update(_material_rgba_from_mjcf(resolved))

    return material_colors


def _material_has_surface_shader(material) -> bool:
    if not getattr(material, "use_nodes", False) or material.node_tree is None:
        return False
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeOutputMaterial":
            continue
        surface = node.inputs.get("Surface")
        if surface is not None and surface.links:
            return True
    return False


def _is_transmissive_rgba(rgba: tuple[float, float, float, float]) -> bool:
    return rgba[3] < 1.0 - _TRANSMISSIVE_ALPHA_EPSILON


def _ior_for_transmissive_material(name: str) -> float:
    key = _normalized_material_name(name).lower()
    if "water" in key:
        return 1.333
    if "acrylic" in key or "plexiglass" in key:
        return 1.49
    if "wine" in key or "liquid" in key:
        return 1.36
    return 1.45


def _roughness_for_transmissive_material(name: str) -> float:
    key = _normalized_material_name(name).lower()
    if "frosted" in key or "lens" in key:
        return 0.18
    if "smoked" in key:
        return 0.04
    if "water" in key or "wine" in key or "liquid" in key:
        return 0.0
    return 0.01


def _set_principled_input(node, input_name: str, value: float | tuple[float, float, float, float]) -> None:
    if input_name in node.inputs:
        node.inputs[input_name].default_value = value


def _apply_principled_material_inputs(node, material_name: str, rgba: tuple[float, float, float, float]) -> None:
    transmissive = _is_transmissive_rgba(rgba)
    shader_rgba = (rgba[0], rgba[1], rgba[2], 1.0) if transmissive else rgba
    _set_principled_input(node, "Base Color", shader_rgba)
    _set_principled_input(node, "Roughness", _roughness_for_transmissive_material(material_name) if transmissive else 0.55)
    _set_principled_input(node, "Metallic", 0.0)
    _set_principled_input(node, "Alpha", shader_rgba[3])
    _set_principled_input(node, "Transmission Weight", 1.0 if transmissive else 0.0)
    if transmissive:
        _set_principled_input(node, "IOR", _ior_for_transmissive_material(material_name))
        _set_principled_input(node, "Specular IOR Level", 1.0)


def _repair_material_surface_shader(material, rgba: tuple[float, float, float, float] | None = None) -> None:
    rgba = rgba if rgba is not None else _rgba_from_material(material)
    transmissive = _is_transmissive_rgba(rgba)
    material_rgba = (rgba[0], rgba[1], rgba[2], 1.0) if transmissive else rgba
    if _material_has_surface_shader(material):
        if rgba is not None:
            for node in material.node_tree.nodes:
                if node.bl_idname != "ShaderNodeBsdfPrincipled":
                    continue
                _apply_principled_material_inputs(node, material.name, rgba)
        material.diffuse_color = material_rgba
        material.blend_method = "OPAQUE"
        if hasattr(material, "use_screen_refraction"):
            material.use_screen_refraction = transmissive
        if hasattr(material, "show_transparent_back"):
            material.show_transparent_back = False
        return
    material.use_nodes = True
    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    _apply_principled_material_inputs(bsdf, material.name, rgba)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    material.diffuse_color = material_rgba
    material.blend_method = "OPAQUE"
    if hasattr(material, "use_screen_refraction"):
        material.use_screen_refraction = transmissive
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False


def _new_material_from_rgba(name: str, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name=name)
    _repair_material_surface_shader(material, rgba)
    return material


def _float_tuple(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(value) for value in text.split())


def _mjcf_local_quat(element: ET.Element) -> Quaternion:
    quat = _float_tuple(element.get("quat"), (1.0, 0.0, 0.0, 0.0))
    return Quaternion((quat[0], quat[1], quat[2], quat[3]))


def _set_local_transform(obj, pos: tuple[float, ...], quat: Quaternion) -> None:
    obj.location = tuple(float(value) for value in pos[:3])
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat


def _normalized_scale(scale) -> tuple[float, float, float]:
    """``RigidObjCfg.scale`` may be scalar or 3-tuple post __post_init__; coerce."""
    if isinstance(scale, (int, float)):
        return (float(scale), float(scale), float(scale))
    if scale is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(s) for s in scale)
    if len(values) == 1:
        return (values[0],) * 3
    if len(values) >= 3:
        return values[:3]
    raise ValueError(f"Cannot interpret scale {scale!r}")


def _pose_matrix(pos: tuple[float, ...], quat: Quaternion) -> Matrix:
    return Matrix.Translation(Vector(tuple(float(value) for value in pos[:3]))) @ quat.to_matrix().to_4x4()


def _scale_tuple(scale) -> tuple[float, float, float]:
    return _normalized_scale(scale)


def _mjcf_material_rgba(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    materials: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall(".//material"):
        name = material.get("name")
        rgba_text = material.get("rgba")
        if name is None or rgba_text is None:
            continue
        rgba = _rgba_from_text(rgba_text)
        if rgba is not None:
            materials[name] = rgba
    return materials


def _mjcf_material_textures(root: ET.Element, mjcf_dir: Path) -> dict[str, Path]:
    """Map MJCF material name → resolved image-texture file path.

    Parses ``<asset><texture name=... file=...>`` and
    ``<asset><material name=... texture=...>`` references. Honours
    ``<compiler texturedir="...">`` and falls back to the MJCF's own
    directory. Returns only entries whose texture file exists on disk.
    """
    compiler = root.find("compiler")
    texdir = compiler.get("texturedir") if compiler is not None else None
    base_dir = (mjcf_dir / texdir).resolve() if texdir else mjcf_dir
    textures: dict[str, Path] = {}
    for texture in root.findall(".//asset/texture"):
        name = texture.get("name")
        filename = texture.get("file")
        if not name or not filename:
            continue
        resolved = _resolve_asset_path(filename, base_dir)
        if resolved is not None:
            textures[name] = resolved
    material_textures: dict[str, Path] = {}
    for material in root.findall(".//asset/material"):
        name = material.get("name")
        tex_ref = material.get("texture")
        if not name or not tex_ref or tex_ref not in textures:
            continue
        material_textures[name] = textures[tex_ref]
    return material_textures


def _apply_texture_to_material(material, texture_path: Path) -> None:
    """Add an Image Texture node to ``material`` and route it to Base Color.

    Idempotent — re-applying replaces the existing image. Uses UV coords from
    the mesh's active UV map.
    """
    material.use_nodes = True
    nt = material.node_tree
    bsdf = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )
    if bsdf is None:
        return
    tex_node = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeTexImage"),
        None,
    )
    if tex_node is None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (-300, 300)
    tex_node.image = bpy.data.images.load(str(texture_path), check_existing=True)
    uv_node = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeUVMap"),
        None,
    )
    if uv_node is None:
        uv_node = nt.nodes.new("ShaderNodeUVMap")
        uv_node.location = (-500, 300)
    nt.links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])


def _dedupe_overlapping_geoms(
    geoms: list[ET.Element], geom_defaults: dict[str, dict[str, str]]
) -> list[ET.Element]:
    """Drop duplicate geoms at the same (type, size, pos, quat) within a body.

    URDF→MJCF converters (notably the one RLBench uses for close_box) emit
    two stacked geoms per face: a named one carrying a class-default material
    (often a placeholder colour like the ``visualgeom`` green), and an
    anonymous one with an explicit ``rgba``. Both end up at group=1 and
    z-fight in Cycles. We keep one per signature, preferring the variant
    with an explicit ``rgba``, and deprioritising class-default ``material``
    assignments where the material name matches the class name — that's the
    classic "this is just a placeholder colour" pattern.
    """
    def key(geom):
        a = _mjcf_geom_resolved_attrs(geom, geom_defaults)
        return (
            (a.get("type") or "sphere").lower(),
            a.get("size", ""),
            a.get("pos", ""),
            a.get("quat", ""),
            (a.get("mesh") or ""),
        )

    def score(geom):
        # Higher = better (kept).
        own = geom.attrib  # attributes set ON the geom itself
        a = _mjcf_geom_resolved_attrs(geom, geom_defaults)
        s = 0
        rgba_text = (own.get("rgba") or "").strip()
        if rgba_text:
            parts = rgba_text.split()
            if len(parts) >= 4:
                try:
                    if float(parts[3]) > 0.0:
                        s += 10
                except ValueError:
                    pass
            else:
                s += 5
        if "material" in own:
            # Material set directly on the geom — strong intent.
            s += 3
        else:
            class_name = own.get("class")
            inherited_mat = a.get("material")
            if class_name and inherited_mat and inherited_mat == class_name:
                # Class default placeholder ("visualgeom"-style) — actively
                # penalise so an anonymous unstyled sibling wins.
                s -= 2
            elif inherited_mat:
                s += 1
        name = own.get("name", "")
        if not (name.endswith("_collision") or name.endswith("_col")):
            s += 1
        return s

    by_key: dict[tuple, ET.Element] = {}
    for geom in geoms:
        k = key(geom)
        prev = by_key.get(k)
        if prev is None or score(geom) > score(prev):
            by_key[k] = geom
    kept_ids = {id(g) for g in by_key.values()}
    return [g for g in geoms if id(g) in kept_ids]


def _is_mjcf_collision_only_geom(attrs: dict[str, str]) -> bool:
    """Heuristic: is this geom collision-only and should be skipped at render time.

    MJCFs use several conventions to mark collision-only geoms — there isn't
    one canonical flag. We treat any of these as collision:
      * ``group="3"`` — mujoco's default split (franka style)
      * ``class`` containing "collision" — explicit class naming
      * ``rgba`` with alpha 0 — author flagged invisible (libero / open6dor style)
      * ``name`` ending in ``_collision`` — naming convention from URDF→MJCF
        converters (RLBench close_box-style: each face has a ``*_collision``
        geom overlapping a plain rgba geom, both technically group="1",
        author intent being that the unnamed one is the visual representation)

    All four checks together let us strip the collision shells from real
    asset packs (libero, robosuite, open6dor, RLBench MJCFs) without losing
    legitimate visuals.
    """
    if attrs.get("group") == "3":
        return True
    class_name = attrs.get("class", "")
    if class_name and "collision" in class_name.lower():
        return True
    name = attrs.get("name", "")
    if name.endswith("_collision") or name.endswith("_col"):
        return True
    rgba_text = attrs.get("rgba")
    if rgba_text:
        parts = rgba_text.split()
        if len(parts) >= 4:
            try:
                if float(parts[3]) <= 0.0:
                    return True
            except ValueError:
                pass
    return False


def _mjcf_geom_default_attrs(root: ET.Element) -> dict[str, dict[str, str]]:
    """Resolve ``<default class="..."><geom .../></default>`` attribute inheritance.

    MJCF lets geoms inherit attributes from named default classes — e.g.
    franka's ``<default class="visual"><geom type="mesh" ... /></default>``
    means any geom with ``class="visual"`` and no own ``type`` attribute is
    a mesh. Without this expansion such geoms silently fall back to spheres.

    Nested defaults (``<default class="panda"><default class="visual">...</default></default>``)
    inherit from their lexical parent. We walk the tree recursively, accumulating
    geom attributes class-by-class. Returns ``{class_name: {attr: value}}``.
    """
    result: dict[str, dict[str, str]] = {}

    def walk(element: ET.Element, inherited: dict[str, str]) -> None:
        own = inherited.copy()
        for geom_default in element.findall("geom"):
            for attr, value in geom_default.attrib.items():
                own[attr] = value
        class_name = element.get("class")
        if class_name is not None:
            result[class_name] = own
        for child in element.findall("default"):
            walk(child, own)

    for top in root.findall("default"):
        walk(top, {})
    return result


def _mjcf_geom_resolved_attrs(
    geom: ET.Element, defaults: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Merge a geom's own attrs over its class default attrs."""
    class_name = geom.get("class")
    merged: dict[str, str] = dict(defaults.get(class_name, {})) if class_name else {}
    merged.update(geom.attrib)
    return merged


def _mjcf_mesh_assets(
    root: ET.Element, mjcf_dir: Path
) -> dict[str, tuple[Path, tuple[float, float, float]]]:
    """Parse ``<asset><mesh name=... file=... scale=...>`` references.

    Honours ``<compiler meshdir="...">`` for path resolution. Returns a map
    of asset name → (resolved absolute path, xyz scale). Entries whose file
    cannot be resolved are dropped (caller logs / errors elsewhere).
    """
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    base_dir = (mjcf_dir / meshdir).resolve() if meshdir else mjcf_dir
    assets: dict[str, tuple[Path, tuple[float, float, float]]] = {}
    for mesh in root.findall(".//asset/mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        # MJCF spec: if ``name`` is omitted the mesh's name defaults to the
        # file's stem (e.g. ``<mesh file="link0_0.obj"/>`` → name "link0_0").
        name = mesh.get("name") or Path(filename).stem
        resolved = _resolve_asset_path(filename, base_dir)
        if resolved is None:
            continue
        scale = _float_tuple(mesh.get("scale"), (1.0, 1.0, 1.0))
        if len(scale) < 3:
            scale = (scale[0],) * 3
        assets[name] = (resolved, (float(scale[0]), float(scale[1]), float(scale[2])))
    return assets


def _mjcf_geom_material(
    geom: ET.Element,
    materials: dict[str, tuple[float, float, float, float]],
    geom_defaults: dict[str, dict[str, str]] | None = None,
) -> tuple[tuple[float, float, float, float], str | None]:
    attrs = _mjcf_geom_resolved_attrs(geom, geom_defaults or {})
    material_name = attrs.get("material")
    if material_name is not None and material_name in materials:
        return materials[material_name], material_name
    rgba_text = attrs.get("rgba")
    if rgba_text:
        rgba = _rgba_from_text(rgba_text)
        if rgba is not None:
            return rgba, material_name
    return _rgba_srgb_to_linear((0.7, 0.7, 0.7, 1.0)), material_name


def _is_solid_transmissive_volume(name: str) -> bool:
    key = _normalized_material_name(name).lower()
    return any(token in key for token in ("water", "wine", "liquid"))


def _transmissive_bevel_amount(size: tuple[float, ...]) -> float:
    min_extent = min(abs(float(value)) for value in size if abs(float(value)) > 0.0)
    return min(max(min_extent * 0.08, 0.0006), 0.004)


def _polish_transmissive_visual(obj, *, bevel_amount: float) -> None:
    if getattr(obj, "type", None) != "MESH":
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    bevel = obj.modifiers.new(name="metasim_transmissive_bevel", type="BEVEL")
    bevel.width = bevel_amount
    bevel.segments = 3
    if hasattr(bevel, "harden_normals"):
        bevel.harden_normals = True
    weighted_normal = obj.modifiers.new(name="metasim_transmissive_weighted_normals", type="WEIGHTED_NORMAL")
    if hasattr(weighted_normal, "keep_sharp"):
        weighted_normal.keep_sharp = True


def _create_hollow_cylinder_visual(name: str, *, radius: float, half_height: float):
    segments = 96
    wall_thickness = min(max(radius * 0.08, 0.0015), radius * 0.25)
    bottom_thickness = min(max(half_height * 0.08, 0.002), half_height * 0.25)
    inner_radius = max(radius - wall_thickness, radius * 0.4)
    bottom_z = -half_height
    inner_bottom_z = bottom_z + bottom_thickness
    top_z = half_height

    vertices: list[tuple[float, float, float]] = []
    for ring_radius, z in (
        (radius, bottom_z),
        (radius, top_z),
        (inner_radius, inner_bottom_z),
        (inner_radius, top_z),
    ):
        for index in range(segments):
            angle = tau * index / segments
            vertices.append((ring_radius * cos(angle), ring_radius * sin(angle), z))

    outer_bottom = 0
    outer_top = segments
    inner_bottom = 2 * segments
    inner_top = 3 * segments
    faces: list[tuple[int, ...]] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append((outer_bottom + index, outer_bottom + next_index, outer_top + next_index, outer_top + index))
        faces.append((inner_bottom + next_index, inner_bottom + index, inner_top + index, inner_top + next_index))
        faces.append((outer_top + index, outer_top + next_index, inner_top + next_index, inner_top + index))
        faces.append((outer_bottom + next_index, outer_bottom + index, inner_bottom + index, inner_bottom + next_index))
    faces.append(tuple(reversed(range(outer_bottom, outer_bottom + segments))))
    faces.append(tuple(range(inner_bottom, inner_bottom + segments)))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _create_mjcf_geom_visual(
    geom: ET.Element,
    *,
    body_name: str,
    material_rgba,
    material_name: str | None,
    parent,
    mesh_assets: dict[str, tuple[Path, tuple[float, float, float]]] | None = None,
    geom_defaults: dict[str, dict[str, str]] | None = None,
    material_textures: dict[str, Path] | None = None,
):
    attrs = _mjcf_geom_resolved_attrs(geom, geom_defaults or {})
    geom_type = (attrs.get("type") or "sphere").lower()
    size = _float_tuple(attrs.get("size"), (0.05,))
    name = attrs.get("name") or f"{body_name}_{geom_type}"
    pos = _float_tuple(attrs.get("pos"), (0.0, 0.0, 0.0))
    quat = Quaternion(_float_tuple(attrs.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    transmissive = _is_transmissive_rgba(material_rgba)
    visual_name = f"{name} {material_name or ''}"

    if geom_type == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.dimensions = (2.0 * size[0], 2.0 * size[1], 2.0 * size[2])
        bpy.context.view_layer.update()
    elif geom_type == "cylinder":
        if transmissive and not _is_solid_transmissive_volume(visual_name):
            obj = _create_hollow_cylinder_visual(name, radius=float(size[0]), half_height=float(size[1]))
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=float(size[0]), depth=2.0 * float(size[1]))
            obj = bpy.context.object
    elif geom_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=float(size[0]))
        obj = bpy.context.object
    elif geom_type == "mesh":
        mesh_ref = attrs.get("mesh")
        if not mesh_ref or mesh_assets is None or mesh_ref not in mesh_assets:
            raise ValueError(
                f"MJCF mesh geom {name!r} references unknown mesh asset {mesh_ref!r}"
            )
        mesh_path, mesh_scale = mesh_assets[mesh_ref]
        obj = import_mesh(str(mesh_path))
        obj.scale = mesh_scale
    else:
        raise ValueError(f"Unsupported MJCF geom type for Blender visual import: {geom_type!r}")

    obj.name = name
    obj.parent = parent
    _set_local_transform(obj, pos, quat)
    material = _new_material_from_rgba(material_name or f"{name}_material", material_rgba)
    if material_name and material_textures and material_name in material_textures:
        _apply_texture_to_material(material, material_textures[material_name])
    _assign_material(obj, material)
    if transmissive:
        _polish_transmissive_visual(obj, bevel_amount=_transmissive_bevel_amount(size))
    return obj


def _assign_material(obj, material) -> None:
    """Replace ``obj``'s first material slot with ``material`` (or append if empty).

    ``import_mesh`` on a ``.obj`` with a missing ``.mtl`` leaves a stub default
    material in slot 0 that all face-loops are bound to — appending another
    slot doesn't help because faces don't re-bind. Replacing the slot does.
    """
    if getattr(obj, "data", None) is None or not hasattr(obj.data, "materials"):
        return
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def _create_compiled_mjcf_mesh_visual(
    *,
    model,
    mesh_id: int,
    name: str,
    parent,
    geom_pos: tuple[float, ...],
    geom_quat: Quaternion,
    material_rgba: tuple[float, float, float, float],
    material_name: str | None,
    material_textures: dict[str, Path] | None = None,
):
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = [
        tuple(float(value) for value in vertex)
        for vertex in model.mesh_vert[vert_adr : vert_adr + vert_num]
    ]
    faces = [
        tuple(int(value) for value in face)
        for face in model.mesh_face[face_adr : face_adr + face_num]
    ]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    mesh_quat = Quaternion(tuple(float(value) for value in model.mesh_quat[mesh_id]))
    obj.matrix_basis = _pose_matrix(geom_pos, geom_quat) @ _pose_matrix(tuple(model.mesh_pos[mesh_id]), mesh_quat)
    material = _new_material_from_rgba(material_name or f"{name}_material", material_rgba)
    if material_name and material_textures and material_name in material_textures:
        _apply_texture_to_material(material, material_textures[material_name])
    _assign_material(obj, material)
    return obj


def _import_mjcf_visual_tree(
    mjcf_path: str | Path,
    *,
    root_name: str,
    default_position: tuple[float, float, float],
    default_orientation: tuple[float, float, float, float],
    default_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[object, dict[str, object]]:
    mjcf_path = Path(mjcf_path)
    root = ET.parse(mjcf_path).getroot()
    materials = _mjcf_material_rgba(root)
    mesh_assets = _mjcf_mesh_assets(root, mjcf_path.parent)
    geom_defaults = _mjcf_geom_default_attrs(root)
    material_textures = _mjcf_material_textures(root, mjcf_path.parent)
    compiled_model = None
    mesh_name_to_id: dict[str, int] = {}
    mesh_refs = {
        attrs["mesh"]
        for geom in root.findall(".//geom")
        for attrs in (_mjcf_geom_resolved_attrs(geom, geom_defaults),)
        if attrs.get("mesh")
    }
    if mesh_refs:
        try:
            import mujoco
        except ImportError:
            compiled_model = None
        else:
            compiled_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            mesh_name_to_id = {
                mujoco.mj_id2name(compiled_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id): mesh_id
                for mesh_id in range(compiled_model.nmesh)
            }
    root_empty = bpy.data.objects.new(root_name, None)
    bpy.context.collection.objects.link(root_empty)
    _set_local_transform(root_empty, default_position, Quaternion(default_orientation))
    root_empty.scale = tuple(float(s) for s in default_scale)

    body_objects: dict[str, object] = {}
    # Mirror mujoco's convention: unnamed ``<body>`` elements get
    # ``unnamed_body_<N>`` where N counts unnamed bodies across the model.
    # This matters because the physics handler reports body states by these
    # auto-generated names, so the Blender side has to register them under
    # the same names for ``_apply_body_state`` to find them.
    unnamed_counter = [0]

    def visit_body(body: ET.Element, parent) -> None:
        body_name = body.get("name")
        if not body_name:
            body_name = f"unnamed_body_{unnamed_counter[0]}"
            unnamed_counter[0] += 1
        body_parent = parent
        body_empty = bpy.data.objects.new(body_name, None)
        bpy.context.collection.objects.link(body_empty)
        body_empty.parent = parent
        _set_local_transform(
            body_empty,
            _float_tuple(body.get("pos"), (0.0, 0.0, 0.0)),
            _mjcf_local_quat(body),
        )
        body_objects[body_name] = body_empty
        body_parent = body_empty

        geom_list = _dedupe_overlapping_geoms(body.findall("geom"), geom_defaults)
        for geom in geom_list:
            attrs = _mjcf_geom_resolved_attrs(geom, geom_defaults)
            if _is_mjcf_collision_only_geom(attrs):
                continue
            material_rgba, material_name = _mjcf_geom_material(geom, materials, geom_defaults)
            mesh_name = attrs.get("mesh")
            if mesh_name and compiled_model is not None:
                mesh_id = mesh_name_to_id.get(mesh_name)
                if mesh_id is None:
                    raise ValueError(f"MJCF mesh geom {geom.get('name') or mesh_name!r} references unknown mesh {mesh_name!r}")
                _create_compiled_mjcf_mesh_visual(
                    model=compiled_model,
                    mesh_id=mesh_id,
                    name=attrs.get("name") or f"{body_name or root_name}_mesh",
                    parent=body_parent,
                    geom_pos=_float_tuple(attrs.get("pos"), (0.0, 0.0, 0.0)),
                    geom_quat=Quaternion(_float_tuple(attrs.get("quat"), (1.0, 0.0, 0.0, 0.0))),
                    material_rgba=material_rgba,
                    material_name=material_name,
                    material_textures=material_textures,
                )
                continue
            _create_mjcf_geom_visual(
                geom,
                body_name=body_name or root_name,
                material_rgba=material_rgba,
                material_name=material_name,
                parent=body_parent,
                mesh_assets=mesh_assets,
                geom_defaults=geom_defaults,
                material_textures=material_textures,
            )
        for child in body.findall("body"):
            visit_body(child, body_parent)

    worldbody = root.find("worldbody")
    # Some MJCFs are pure asset libraries with no ``<worldbody>`` (e.g. the
    # humanoid_bench tunnel.xml referenced by h1.crawl). Treat that as an
    # empty visual tree rather than an error so the surrounding pipeline
    # keeps going.
    if worldbody is not None:
        for body in worldbody.findall("body"):
            visit_body(body, root_empty)

    bpy.context.view_layer.update()
    return root_empty, body_objects


def _repair_imported_materials(
    imported: list,
    asset_material_rgba: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    seen_objects: set[str] = set()
    seen_collections: set[str] = set()
    seen_materials: set[str] = set()
    asset_material_rgba = asset_material_rgba or {}

    def visit_object(obj) -> None:
        if obj.name in seen_objects:
            return
        seen_objects.add(obj.name)
        if getattr(obj, "type", None) == "MESH":
            if not obj.material_slots and len(asset_material_rgba) == 1:
                material_name, rgba = next(iter(asset_material_rgba.items()))
                obj.data.materials.append(_new_material_from_rgba(material_name, rgba))
            for slot in obj.material_slots:
                material = slot.material
                if material is None:
                    continue
                if material.name in seen_materials:
                    continue
                seen_materials.add(material.name)
                rgba = next(
                    (
                        asset_material_rgba[name]
                        for name in _material_name_candidates(material.name)
                        if name in asset_material_rgba
                    ),
                    None,
                )
                _repair_material_surface_shader(material, rgba)
        collection = getattr(obj, "instance_collection", None)
        if collection is not None and collection.name not in seen_collections:
            seen_collections.add(collection.name)
            for child in collection.objects:
                visit_object(child)
        for child in getattr(obj, "children", ()):
            visit_object(child)

    for obj in imported:
        visit_object(obj)


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
        self._scene_objs: list[object] = []
        self._camera_objs: dict[str, object] = {}
        self._last_camera_states: dict[str, CameraState] = {}
        self._last_tensor_state: TensorState | None = None
        # Dirty-tracks whether the active scene transforms have changed since
        # the camera images in ``_last_camera_states`` were rendered. ``True``
        # means the next ``_get_states`` must re-render before returning, so
        # the first frame after ``set_states`` reflects the requested pose
        # rather than the launch-time defaults.
        self._render_dirty: bool = True
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="metasim_blender_"))

    def launch(self) -> None:
        super().launch()
        self._clear_scene()
        self._configure_render()
        self._import_scene()
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
        render_cfg = getattr(self.scenario, "render", None)
        scene.render.engine = "CYCLES"
        settings = _CYCLES_FINAL_RENDER_SETTINGS
        scene.cycles.samples = int(getattr(render_cfg, "samples", None) or settings["samples"])
        selected_device = self._configure_cycles_device(str(getattr(render_cfg, "device", "AUTO") or "AUTO"))
        scene.render.use_persistent_data = True
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        scene.render.image_settings.file_format = _BLENDER_RGB_READBACK_FORMAT
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "denoising_use_gpu"):
            scene.cycles.denoising_use_gpu = selected_device != "CPU"
        for attr in ("denoising_prefilter", "denoising_quality"):
            if hasattr(scene.cycles, attr):
                setattr(scene.cycles, attr, settings[attr])
        for attr, value in (
            ("max_bounces", settings["max_bounces"]),
            ("diffuse_bounces", settings["diffuse_bounces"]),
            ("glossy_bounces", settings["glossy_bounces"]),
            ("transmission_bounces", settings["transmission_bounces"]),
            ("transparent_max_bounces", settings["transparent_max_bounces"]),
            ("caustics_reflective", settings["caustics_reflective"]),
            ("caustics_refractive", settings["caustics_refractive"]),
            ("use_transparent_shadows", True),
        ):
            if hasattr(scene.cycles, attr):
                setattr(scene.cycles, attr, value)
        try:
            scene.view_settings.view_transform = "Standard"
        except TypeError:
            pass
        scene.view_settings.exposure = -0.55
        scene.view_settings.gamma = 1.0
        scene.render.film_transparent = False

    def _configure_cycles_device(self, device: str) -> str:
        scene = self.context.scene
        normalized = device.upper()
        if normalized == "CPU":
            scene.cycles.device = "CPU"
            return "CPU"
        if normalized == "AUTO":
            for candidate in _CYCLES_GPU_DEVICE_PREFERENCE:
                if self._try_enable_cycles_device(candidate):
                    scene.cycles.device = "GPU"
                    return candidate
            scene.cycles.device = "CPU"
            return "CPU"

        if self._try_enable_cycles_device(normalized):
            scene.cycles.device = "GPU"
            return normalized
        raise RuntimeError(f"Requested Blender/Cycles device {normalized!r} is not available")

    def _try_enable_cycles_device(self, device: str) -> bool:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = device
            prefs.get_devices()
        except TypeError:
            return False
        enabled = 0
        for blender_device in prefs.devices:
            use_device = blender_device.type == device
            blender_device.use = use_device
            enabled += int(use_device)
        return enabled > 0

    def _import_scene(self) -> None:
        scene_cfg = getattr(self.scenario, "scene", None)
        if scene_cfg is None:
            return
        scene_path = scene_cfg.file_name("blender")
        if not scene_path:
            raise ValueError(f"Scene {scene_cfg.name!r} has no Blender asset path")
        result = import_usd_visuals(
            scene_path,
            root_name=scene_cfg.name,
            default_position=getattr(scene_cfg, "default_position", None) or (0.0, 0.0, 0.0),
            default_orientation=_scene_orientation_wxyz(scene_cfg),
            scale=_scale_tuple(getattr(scene_cfg, "scale", (1.0, 1.0, 1.0))),
        )
        result.root.matrix_basis = _scene_usd_xform_matrix(scene_cfg)
        _apply_optional_usd_stage_enhancements(scene_path, result.imported)
        self._scene_objs.append(result.root)

    def _add_lights(self) -> None:
        scene = self.context.scene
        self._setup_world(scene)
        add_scenario_lights(list(getattr(self.scenario, "lights", []) or []))
        self._add_default_ground(scene)

    def _setup_world(self, scene) -> None:
        """World environment lighting.

        Priority:
          1. ``scenario.render.hdri_path`` set → image-based lighting (random
             pick if a directory).
          2. Otherwise a procedural sky gradient (warm top, cool ground) so
             renders have varied reflections without requiring asset files.

        Any external scene-augmentation tool (e.g.
        ``metasim.sim.blender.utils.scene_aug.setup_world_hdri``) can rewrite
        the world node graph after launch to replace either default.
        """
        if scene.world is None:
            scene.world = bpy.data.worlds.new("metasim_world")
        render_cfg = getattr(self.scenario, "render", None)
        hdri_path = getattr(render_cfg, "hdri_path", None) if render_cfg else None
        if hdri_path and self._setup_world_from_hdri(scene, hdri_path):
            return
        scene.world.color = (0.18, 0.20, 0.24)
        scene.world.use_nodes = True
        nt = scene.world.node_tree
        nt.nodes.clear()
        output = nt.nodes.new("ShaderNodeOutputWorld")
        output.location = (400, 0)
        background = nt.nodes.new("ShaderNodeBackground")
        background.location = (200, 0)
        background.inputs["Strength"].default_value = 0.85
        gradient_mix = nt.nodes.new("ShaderNodeMixRGB")
        gradient_mix.location = (0, 0)
        gradient_mix.inputs["Color1"].default_value = (0.42, 0.48, 0.56, 1.0)  # cool ground glow
        gradient_mix.inputs["Color2"].default_value = (0.78, 0.74, 0.66, 1.0)  # warm sky glow
        gradient = nt.nodes.new("ShaderNodeValToRGB")
        gradient.location = (-220, 0)
        gradient.color_ramp.elements[0].position = 0.32
        gradient.color_ramp.elements[0].color = (0.16, 0.18, 0.22, 1.0)
        gradient.color_ramp.elements[1].position = 0.65
        gradient.color_ramp.elements[1].color = (0.78, 0.74, 0.66, 1.0)
        separate = nt.nodes.new("ShaderNodeSeparateXYZ")
        separate.location = (-440, 0)
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-660, 0)
        nt.links.new(tex_coord.outputs["Generated"], separate.inputs["Vector"])
        nt.links.new(separate.outputs["Z"], gradient.inputs["Fac"])
        nt.links.new(gradient.outputs["Color"], gradient_mix.inputs["Fac"])
        nt.links.new(gradient_mix.outputs["Color"], background.inputs["Color"])
        nt.links.new(background.outputs["Background"], output.inputs["Surface"])

    def _setup_world_from_hdri(self, scene, hdri_path: str) -> bool:
        """Load ``hdri_path`` as image-based world lighting. ``hdri_path`` may
        be a file or a directory of .hdr/.exr files (random pick).
        Returns True on success."""
        import random as _random

        path = Path(hdri_path)
        candidates: list[Path] = []
        if path.is_dir():
            for ext in (".hdr", ".exr", ".HDR", ".EXR"):
                candidates.extend(path.rglob(f"*{ext}"))
        elif path.is_file():
            candidates = [path]
        if not candidates:
            return False
        chosen = _random.choice(sorted(candidates))

        scene.world.use_nodes = True
        nt = scene.world.node_tree
        nt.nodes.clear()
        output = nt.nodes.new("ShaderNodeOutputWorld")
        output.location = (400, 0)
        background = nt.nodes.new("ShaderNodeBackground")
        background.location = (200, 0)
        background.inputs["Strength"].default_value = 1.0
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.location = (-100, 0)
        env.image = bpy.data.images.load(str(chosen), check_existing=True)
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.location = (-300, 0)
        mapping.inputs["Rotation"].default_value[2] = _random.uniform(-3.14159, 3.14159)
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-500, 0)
        nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], background.inputs["Color"])
        nt.links.new(background.outputs["Background"], output.inputs["Surface"])
        return True

    def _add_default_ground(self, scene) -> None:
        """Add a 4×4m neutral ground plane *just below* z=0 so renders have a
        floor instead of objects floating in a void. The slight downward
        offset avoids z-fighting when callers add their own ground at z=0
        (e.g. ``scene_aug.add_ground_plane`` in the photoreal demo). To opt
        out: delete the ``metasim_ground`` object after handler.launch().

        Also adds a ``metasim_table`` plane at z=0 — a smaller, warmer
        tabletop where manipulation tasks place their objects (most
        roboverse scenarios assume z=0 *is* the working surface, not the
        floor). The floor stays underneath to fill out the periphery and
        give HDRI lighting something to bounce off.
        """
        if bpy.data.objects.get("metasim_ground") is None:
            bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, -0.011))
            plane = bpy.context.object
            plane.name = "metasim_ground"
            material = bpy.data.materials.new("metasim_ground_material")
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (0.20, 0.20, 0.22, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.92
                if "Specular IOR Level" in bsdf.inputs:
                    bsdf.inputs["Specular IOR Level"].default_value = 0.35
            plane.data.materials.append(material)

        if bpy.data.objects.get("metasim_table") is None:
            self._add_default_table(scene)

    def _add_default_table(self, scene) -> None:
        """Wooden tabletop at z=0 (~1.6×1.2 m), warm tan with subtle noise.

        Sits ~10 mm above the floor, slightly larger than typical
        manipulation workspaces so robot bases + scene objects all rest on
        a believable surface instead of a featureless plane.

        Opt out: delete the ``metasim_table`` object after
        ``handler.launch()`` (its presence is checked but never enforced).
        """
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.005))
        table = bpy.context.object
        table.name = "metasim_table"
        # Box scaled to 1.6 m × 1.2 m × 0.01 m (a thin slab so the top is at z≈0).
        table.scale = (1.6, 1.2, 0.01)
        bpy.context.view_layer.objects.active = table
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        material = bpy.data.materials.new("metasim_table_material")
        material.use_nodes = True
        nt = material.node_tree
        nt.nodes.clear()
        output = nt.nodes.new("ShaderNodeOutputMaterial")
        output.location = (400, 0)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (200, 0)
        bsdf.inputs["Base Color"].default_value = (0.42, 0.30, 0.20, 1.0)  # warm walnut tan
        bsdf.inputs["Roughness"].default_value = 0.55
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.5
        # Subtle noise mixed into the base color → grain hint without an asset.
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.location = (-300, 0)
        noise.inputs["Scale"].default_value = 18.0
        noise.inputs["Detail"].default_value = 2.5
        noise.inputs["Roughness"].default_value = 0.55
        color_ramp = nt.nodes.new("ShaderNodeValToRGB")
        color_ramp.location = (-80, 0)
        color_ramp.color_ramp.elements[0].color = (0.32, 0.22, 0.14, 1.0)
        color_ramp.color_ramp.elements[1].color = (0.55, 0.39, 0.26, 1.0)
        nt.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
        nt.links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        table.data.materials.append(material)

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
        scale = _normalized_scale(obj_cfg.scale)
        blender_path = obj_cfg.file_name("blender")
        suffix = Path(blender_path).suffix.lower() if blender_path else ""
        if blender_path and suffix in MJCF_SUFFIXES:
            root, _ = _import_mjcf_visual_tree(
                blender_path,
                root_name=obj_cfg.name,
                default_position=obj_cfg.default_position,
                default_orientation=obj_cfg.default_orientation,
                default_scale=scale,
            )
            return root
        if blender_path and suffix in USD_SUFFIXES:
            result = import_usd_visuals(
                blender_path,
                root_name=obj_cfg.name,
                default_position=obj_cfg.default_position,
                default_orientation=obj_cfg.default_orientation,
                scale=scale,
            )
            _repair_imported_materials(result.imported, _asset_material_rgba(obj_cfg))
            _apply_optional_usd_stage_enhancements(blender_path, result.imported)
            return result.root
        if obj_cfg.mesh_path is not None:
            obj = import_mesh(obj_cfg.mesh_path)
            obj.name = obj_cfg.name
            obj.scale = scale
            return obj
        raise ValueError(
            f"Rigid object {obj_cfg.name!r} cannot be imported by Blender: "
            f"file_type={obj_cfg.file_type.get('blender')!r}, path={blender_path!r}, "
            "supported suffixes are .usd/.usda/.usdc/.xml/.mjcf or mesh_path"
        )

    def _import_articulation(self, obj_cfg: ArticulationObjCfg) -> None:
        usd_path = obj_cfg.file_name("blender")
        if not usd_path:
            raise ValueError(f"{type(obj_cfg).__name__} {obj_cfg.name!r} requires usd_path for Blender")
        scale = _normalized_scale(getattr(obj_cfg, "scale", 1.0))
        suffix = Path(usd_path).suffix.lower()
        if suffix in MJCF_SUFFIXES:
            root, body_objects = _import_mjcf_visual_tree(
                usd_path,
                root_name=obj_cfg.name,
                default_position=obj_cfg.default_position,
                default_orientation=obj_cfg.default_orientation,
                default_scale=scale,
            )
            self._objs[obj_cfg.name] = root
            self._body_objs[obj_cfg.name] = body_objects
            self._body_names[obj_cfg.name] = sorted(body_objects)
            return
        if suffix not in USD_SUFFIXES:
            raise ValueError(
                f"{type(obj_cfg).__name__} {obj_cfg.name!r} cannot be imported by Blender: "
                f"path={usd_path!r}, supported suffixes are .usd/.usda/.usdc/.xml/.mjcf"
            )
        asset_material_rgba = _asset_material_rgba(obj_cfg)
        result = import_usd_visuals(
            usd_path,
            root_name=obj_cfg.name,
            default_position=obj_cfg.default_position,
            default_orientation=obj_cfg.default_orientation,
            scale=scale,
        )
        _repair_imported_materials(result.imported, asset_material_rgba)
        _apply_optional_usd_stage_enhancements(usd_path, result.imported)
        self._objs[obj_cfg.name] = result.root
        self._register_body_objects(obj_cfg.name, result.imported)

    def _register_body_objects(self, obj_name: str, imported: list) -> None:
        bpy.context.view_layer.update()
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
        # Scene transforms changed — invalidate the cached render so the next
        # ``_get_states`` re-renders the cameras instead of returning stale RGB.
        self._render_dirty = True
        self.refresh_render()

    def _apply_tensor_state(self, state: TensorState) -> None:
        for obj_name, obj_state in state.objects.items():
            if obj_name in self._objs:
                _apply_root_state_to_object(self._objs[obj_name], obj_state.root_state[0])
            if obj_name in self._body_objs:
                self._apply_body_state(obj_name, obj_state, use_robot_aliases=False)
        for robot_name, robot_state in state.robots.items():
            self._apply_robot_body_state(robot_name, robot_state)

    def _apply_robot_body_state(self, robot_name: str, robot_state: RobotState) -> None:
        self._apply_body_state(robot_name, robot_state, use_robot_aliases=True)

    def _apply_body_state(self, obj_name: str, obj_state, *, use_robot_aliases: bool) -> None:
        body_map = self._body_objs.get(obj_name, {})
        missing = []
        mapped = []
        for body_index, body_name in enumerate(obj_state.body_names):
            obj = (
                _resolve_robot_body_object(body_map, body_name)
                if use_robot_aliases
                else body_map.get(body_name)
            )
            if obj is None:
                missing.append(body_name)
                continue
            mapped.append((_object_hierarchy_depth(obj), body_index, obj))
        for _, body_index, obj in sorted(mapped, key=lambda item: item[0]):
            scale_override = _robot_body_pose_scale(obj) if use_robot_aliases else None
            _apply_root_state_to_object(obj, obj_state.body_state[0, body_index], scale_override)
        if missing:
            raise ValueError(f"Blender could not map bodies for {obj_name}: {missing[:10]}")

    def _apply_dict_state(self, state: DictEnvState) -> None:
        """Apply root-only dict states; exact articulated replay requires TensorState."""
        for obj_name, obj_state in state["objects"].items():
            if obj_name in self._objs:
                root = torch.zeros(13)
                root[:3] = obj_state["pos"]
                root[3:7] = obj_state["rot"]
                _apply_root_state_to_object(self._objs[obj_name], root)
            elif obj_name in self._body_objs and "body_state" not in obj_state:
                raise ValueError(
                    f"Blender dict state for articulated object {obj_name!r} lacks body_state. "
                    "Use TensorState/body_state for exact articulated replay."
                )

        for robot_name, robot_state in state.get("robots", {}).items():
            if "body_state" not in robot_state:
                raise ValueError(
                    f"Blender dict state for robot {robot_name!r} cannot apply dof_pos directly. "
                    "Use TensorState/body_state captured from a simulator for exact robot replay."
                )

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids not in (None, [0]):
            raise ValueError("BlenderHandler currently supports only one environment")
        if self._render_dirty:
            # Cached cameras are stale (e.g. user just called set_states without a
            # subsequent simulate). Re-render before returning so the caller sees
            # the requested pose, not the launch-time defaults.
            self.refresh_render()
        if self._last_tensor_state is None:
            return TensorState(objects={}, robots={}, cameras=self._last_camera_states)
        return TensorState(
            objects=self._last_tensor_state.objects,
            robots=self._last_tensor_state.robots,
            cameras=self._last_camera_states,
            extras=self._last_tensor_state.extras,
        )

    def get_states(self, env_ids: list[int] | None = None, mode: str = "dict"):
        """Return current state in ``tensor`` or ``dict`` form.

        Blender is render-only — it carries cameras but no independent
        physics state, so ``_get_states`` can return a TensorState with
        empty ``objects``/``robots``. ``state_tensor_to_nested`` assumes
        at least one object or robot exists, so the empty case is handled
        here directly instead.
        """
        tensor_state = self._get_states(env_ids=env_ids)
        if mode == "tensor":
            return tensor_state
        if mode != "dict":
            raise ValueError(f"Unknown state mode {mode!r}")
        if not tensor_state.objects and not tensor_state.robots:
            num_envs = max(1, getattr(self, "num_envs", 1))
            return [{"objects": {}, "robots": {}} for _ in range(num_envs)]
        return state_tensor_to_nested(self, tensor_state)

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
        self._render_dirty = False
        self._invalidate_state_caches()

    def _render_camera(self, camera: PinholeCameraCfg) -> CameraState:
        scene = self.context.scene
        scene.camera = self._camera_objs[camera.name]
        scene.render.resolution_x = int(camera.width)
        scene.render.resolution_y = int(camera.height)
        scene.render.resolution_percentage = 100
        image_path = self._tmp_dir / f"{camera.name}.{_BLENDER_RGB_READBACK_SUFFIX}"
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
