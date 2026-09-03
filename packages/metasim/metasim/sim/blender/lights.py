from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Quaternion

from metasim.scenario.lights import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, SphereLightCfg


def add_scenario_lights(lights: list) -> None:
    if len(lights) == 0:
        _add_default_area_lights()
        return

    for light in lights:
        if isinstance(light, DomeLightCfg):
            _add_dome_light(light)
        elif isinstance(light, DistantLightCfg):
            _add_distant_light(light)
        elif isinstance(light, SphereLightCfg):
            _add_sphere_light(light)
        elif isinstance(light, DiskLightCfg):
            _add_disk_light(light)
        elif isinstance(light, CylinderLightCfg):
            raise NotImplementedError(
                f"Blender light {light.name!r} uses CylinderLightCfg, which has no direct Blender light mapping. "
                "Use DomeLightCfg, DistantLightCfg, SphereLightCfg, or DiskLightCfg for Blender rendering."
            )
        else:
            raise NotImplementedError(f"Blender light {light.name!r} has unsupported type {type(light).__name__}")


def _ensure_world():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("metasim_world")
    return scene.world


def _add_dome_light(light: DomeLightCfg) -> None:
    world = _ensure_world()
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background = nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = tuple(float(c) for c in light.color) + (1.0,)
        background.inputs["Strength"].default_value = float(light.intensity) / 500.0
    if light.texture_file:
        texture_path = Path(light.texture_file)
        if not texture_path.is_file():
            raise FileNotFoundError(f"Dome light HDRI does not exist for {light.name!r}: {texture_path}")
        env = nodes.new(type="ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(texture_path))
        if background is not None:
            world.node_tree.links.new(env.outputs["Color"], background.inputs["Color"])


def _add_distant_light(light: DistantLightCfg) -> None:
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = light.name
    obj.data.energy = float(light.intensity) / 500.0
    obj.data.color = tuple(float(c) for c in light.color)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Quaternion(tuple(float(v) for v in light.quat))


def _add_sphere_light(light: SphereLightCfg) -> None:
    bpy.ops.object.light_add(type="POINT", location=tuple(float(v) for v in light.pos))
    obj = bpy.context.object
    obj.name = light.name
    obj.data.energy = float(light.intensity)
    obj.data.color = tuple(float(c) for c in light.color)
    obj.data.shadow_soft_size = float(light.radius)


def _add_disk_light(light: DiskLightCfg) -> None:
    bpy.ops.object.light_add(type="AREA", location=tuple(float(v) for v in light.pos))
    obj = bpy.context.object
    obj.name = light.name
    obj.data.energy = float(light.intensity)
    obj.data.color = tuple(float(c) for c in light.color)
    obj.data.shape = "DISK"
    obj.data.size = float(light.radius) * 2.0
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = tuple(float(v) for v in light.rot)


def _add_default_area_lights() -> None:
    world = _ensure_world()
    world.color = (0.02, 0.02, 0.025)
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
