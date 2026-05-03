from __future__ import annotations

import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
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


def import_mesh(path):
    before_names = set(bpy.data.objects.keys())
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

    imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
    active = bpy.context.object
    if active in imported:
        return active
    mesh_objects = [obj for obj in imported if getattr(obj, "type", None) == "MESH"]
    if mesh_objects:
        return mesh_objects[0]
    return active


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
    if normalized.endswith("-material"):
        return (normalized, normalized.removesuffix("-material"))
    return (normalized,)


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


def _mjcf_geom_material(
    geom: ET.Element, materials: dict[str, tuple[float, float, float, float]]
) -> tuple[tuple[float, float, float, float], str | None]:
    material_name = geom.get("material")
    if material_name is not None and material_name in materials:
        return materials[material_name], material_name
    rgba_text = geom.get("rgba")
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
):
    geom_type = (geom.get("type") or "sphere").lower()
    size = _float_tuple(geom.get("size"), (0.05,))
    name = geom.get("name") or f"{body_name}_{geom_type}"
    pos = _float_tuple(geom.get("pos"), (0.0, 0.0, 0.0))
    quat = _mjcf_local_quat(geom)
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
    else:
        raise ValueError(f"Unsupported MJCF geom type for Blender visual import: {geom_type!r}")

    obj.name = name
    obj.parent = parent
    _set_local_transform(obj, pos, quat)
    obj.data.materials.append(_new_material_from_rgba(material_name or f"{name}_material", material_rgba))
    if transmissive:
        _polish_transmissive_visual(obj, bevel_amount=_transmissive_bevel_amount(size))
    return obj


def _import_mjcf_visual_tree(
    mjcf_path: str | Path,
    *,
    root_name: str,
    default_position: tuple[float, float, float],
    default_orientation: tuple[float, float, float, float],
) -> tuple[object, dict[str, object]]:
    root = ET.parse(mjcf_path).getroot()
    materials = _mjcf_material_rgba(root)
    root_empty = bpy.data.objects.new(root_name, None)
    bpy.context.collection.objects.link(root_empty)
    _set_local_transform(root_empty, default_position, Quaternion(default_orientation))

    body_objects: dict[str, object] = {}

    def visit_body(body: ET.Element, parent) -> None:
        body_name = body.get("name")
        body_parent = parent
        if body_name:
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

        for geom in body.findall("geom"):
            material_rgba, material_name = _mjcf_geom_material(geom, materials)
            _create_mjcf_geom_visual(
                geom,
                body_name=body_name or root_name,
                material_rgba=material_rgba,
                material_name=material_name,
                parent=body_parent,
            )
        for child in body.findall("body"):
            visit_body(child, body_parent)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF asset {mjcf_path!s} has no worldbody")
    for body in worldbody.findall("body"):
        visit_body(body, root_empty)

    bpy.context.view_layer.update()
    return root_empty, body_objects


def _repair_imported_materials(
    imported: list, asset_material_rgba: dict[str, tuple[float, float, float, float]] | None = None
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

    def _add_lights(self) -> None:
        scene = self.context.scene
        if scene.world is None:
            scene.world = bpy.data.worlds.new("metasim_world")
        scene.world.color = (0.02, 0.02, 0.025)
        for name, location, energy, size in (
            ("metasim_key_area_light", (0.0, -2.2, 3.0), 300.0, 5.0),
            ("metasim_fill_area_light", (-2.2, 0.4, 1.8), 45.0, 5.0),
            ("metasim_rim_area_light", (1.8, 0.6, 2.4), 70.0, 2.4),
        ):
            bpy.ops.object.light_add(type="AREA", location=location)
            light = bpy.context.object
            light.name = name
            light.data.energy = energy
            light.data.size = size

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
        blender_path = obj_cfg.file_name("blender")
        if blender_path and Path(blender_path).suffix.lower() in {".xml", ".mjcf"}:
            root, _ = _import_mjcf_visual_tree(
                blender_path,
                root_name=obj_cfg.name,
                default_position=obj_cfg.default_position,
                default_orientation=obj_cfg.default_orientation,
            )
            return root
        if obj_cfg.mesh_path is None:
            raise ValueError(f"Rigid object {obj_cfg.name!r} requires mesh_path for Blender")
        obj = import_mesh(obj_cfg.mesh_path)
        obj.name = obj_cfg.name
        return obj

    def _import_articulation(self, obj_cfg: ArticulationObjCfg) -> None:
        usd_path = obj_cfg.file_name("blender")
        if not usd_path:
            raise ValueError(f"{type(obj_cfg).__name__} {obj_cfg.name!r} requires usd_path for Blender")
        if Path(usd_path).suffix.lower() in {".xml", ".mjcf"}:
            _, body_objects = _import_mjcf_visual_tree(
                usd_path,
                root_name=obj_cfg.name,
                default_position=obj_cfg.default_position,
                default_orientation=obj_cfg.default_orientation,
            )
            self._body_objs[obj_cfg.name] = body_objects
            self._body_names[obj_cfg.name] = sorted(body_objects)
            return
        asset_material_rgba = _asset_material_rgba(obj_cfg)
        before_names = set(bpy.data.objects.keys())
        result = bpy.ops.wm.usd_import(filepath=str(usd_path))
        if "FINISHED" not in result:
            raise RuntimeError(f"Failed to import USD for {obj_cfg.name!r}: {result}")
        imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
        if not imported:
            raise RuntimeError(f"USD import for {obj_cfg.name!r} produced no Blender objects")
        _repair_imported_materials(imported, asset_material_rgba)
        self._register_body_objects(obj_cfg.name, imported)

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
        self.refresh_render()

    def _apply_tensor_state(self, state: TensorState) -> None:
        for obj_name, obj_state in state.objects.items():
            if obj_name in self._objs:
                _apply_root_state_to_object(self._objs[obj_name], obj_state.root_state[0])
            elif obj_name in self._body_objs:
                self._apply_body_state(obj_name, obj_state, use_robot_aliases=False)
        for robot_name, robot_state in state.robots.items():
            self._apply_robot_body_state(robot_name, robot_state)

    def _apply_robot_body_state(self, robot_name: str, robot_state: RobotState) -> None:
        self._apply_body_state(robot_name, robot_state, use_robot_aliases=True)

    def _apply_body_state(self, obj_name: str, obj_state, *, use_robot_aliases: bool) -> None:
        body_map = self._body_objs.get(obj_name, {})
        missing = []
        for body_index, body_name in enumerate(obj_state.body_names):
            obj = (
                _resolve_robot_body_object(body_map, body_name)
                if use_robot_aliases
                else body_map.get(body_name)
            )
            if obj is None:
                missing.append(body_name)
                continue
            _apply_root_state_to_object(obj, obj_state.body_state[0, body_index], _robot_body_pose_scale(obj))
        if missing:
            raise ValueError(f"Blender could not map bodies for {obj_name}: {missing[:10]}")

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
