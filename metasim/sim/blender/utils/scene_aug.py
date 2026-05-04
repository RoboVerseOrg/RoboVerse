"""World-level scene augmentation: HDRI lighting, ground plane, key lights.

These are mutators on the active Blender scene — call them between renders
to vary the look. They follow the open6dor renderer's structure (env map +
background plane + key light) but rewritten for Blender 5.x's pip ``bpy``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import bpy

WORLD_NODE_BACKGROUND = "RV_Background"
WORLD_NODE_ENVTEX = "RV_EnvTex"
WORLD_NODE_MAPPING = "RV_Mapping"
WORLD_NODE_TEXCOORD = "RV_TexCoord"
WORLD_NODE_OUTPUT = "RV_WorldOutput"


def _ensure_world_nodes() -> bpy.types.NodeTree:
    """Build a deterministic Background ← EnvTex ← Mapping ← TexCoord world graph.

    Uses fixed node names so subsequent calls just update node properties
    instead of re-creating the graph.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.name = WORLD_NODE_OUTPUT
    out.location = (400, 0)

    bg = nt.nodes.new("ShaderNodeBackground")
    bg.name = WORLD_NODE_BACKGROUND
    bg.location = (200, 0)

    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.name = WORLD_NODE_ENVTEX
    env.location = (-100, 0)

    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.name = WORLD_NODE_MAPPING
    mapping.location = (-300, 0)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.name = WORLD_NODE_TEXCOORD
    tex_coord.location = (-500, 0)

    nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return nt


def setup_world_hdri(
    hdri_path: str,
    rotation_z: float = 0.0,
    strength: float = 1.0,
) -> None:
    """Use ``hdri_path`` as image-based lighting (sets RGB ambient + reflections).

    Args:
        hdri_path: absolute path to a .hdr / .exr equirectangular image.
        rotation_z: yaw rotation of the HDRI in radians.
        strength: world background strength multiplier.
    """
    if not os.path.isfile(hdri_path):
        raise FileNotFoundError(hdri_path)

    nt = _ensure_world_nodes()
    env = nt.nodes[WORLD_NODE_ENVTEX]
    bg = nt.nodes[WORLD_NODE_BACKGROUND]
    mapping = nt.nodes[WORLD_NODE_MAPPING]

    img = bpy.data.images.load(hdri_path, check_existing=True)
    env.image = img
    bg.inputs["Strength"].default_value = float(strength)
    mapping.inputs["Rotation"].default_value[2] = float(rotation_z)


def setup_world_color(rgb: tuple[float, float, float] = (0.05, 0.05, 0.05), strength: float = 1.0) -> None:
    """Plain coloured world background — fallback when no HDRI is desired."""
    nt = _ensure_world_nodes()
    env = nt.nodes[WORLD_NODE_ENVTEX]
    env.image = None
    bg = nt.nodes[WORLD_NODE_BACKGROUND]
    bg.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bg.inputs["Strength"].default_value = float(strength)


def add_sun_light(
    name: str = "RV_Sun",
    direction: tuple[float, float, float] = (-0.3, -0.5, -1.0),
    energy: float = 3.0,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    angle_deg: float = 5.0,
) -> bpy.types.Object:
    """Add (or update) a single Sun light to the scene."""
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT" or obj.data.type != "SUN":
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        light_data = bpy.data.lights.new(name=f"{name}_data", type="SUN")
        obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(obj)
    light = obj.data
    light.energy = float(energy)
    light.color = color
    if hasattr(light, "angle"):
        from math import radians

        light.angle = radians(angle_deg)

    # Encode direction as object rotation: -Z axis aligns with direction.
    from mathutils import Vector

    d = Vector(direction).normalized()
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = d.to_track_quat("-Z", "Y")
    obj.location = (0.0, 0.0, 5.0)
    return obj


def add_ground_plane(
    name: str = "RV_Ground",
    size: float = 6.0,
    z: float = 0.0,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    texture_path: str | None = None,
    pass_index: int = 0,
) -> bpy.types.Object:
    """Add (or replace) a flat ground plane with a simple PBR material."""
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    obj.pass_index = pass_index

    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8

    if texture_path is not None and os.path.isfile(texture_path):
        nt = mat.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path, check_existing=True)
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tex_coord.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def list_hdri_files(hdri_dir: str | os.PathLike) -> list[str]:
    """Return absolute paths to all .hdr / .exr files in a directory (recursive)."""
    root = Path(hdri_dir)
    if not root.exists():
        return []
    files = []
    for ext in (".hdr", ".exr", ".HDR", ".EXR"):
        files += [str(p) for p in root.rglob(f"*{ext}")]
    return sorted(files)


def pick_hdri(hdri_dir: str | os.PathLike, rng: random.Random | None = None) -> str | None:
    """Pick one HDRI file at random; returns None if the directory is empty."""
    files = list_hdri_files(hdri_dir)
    if not files:
        return None
    rng = rng or random
    return rng.choice(files)
