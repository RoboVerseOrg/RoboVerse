"""MJCF for the primitive object configs, shared by the MuJoCo and MJX handlers."""

from __future__ import annotations

from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg


def primitive_xml(obj: PrimitiveCubeCfg | PrimitiveCylinderCfg | PrimitiveSphereCfg) -> str:
    """One-body MJCF model for a primitive, with the config's size and total mass.

    MuJoCo sizes are half-extents: a box takes its half sizes and a cylinder its radius and half height, so the
    shape matches the config's full ``size`` / ``height`` like on the other backends. The geom's ``mass`` sets the
    density, so the body weighs ``obj.mass`` instead of MuJoCo's default 1000 kg/m^3 times the volume.
    """
    if isinstance(obj, PrimitiveCubeCfg):
        size_str = f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}"
        type_str = "box"
    elif isinstance(obj, PrimitiveCylinderCfg):
        size_str = f"{obj.radius} {obj.height / 2}"
        type_str = "cylinder"
    elif isinstance(obj, PrimitiveSphereCfg):
        size_str = f"{obj.radius}"
        type_str = "sphere"
    else:
        raise ValueError(f"Unknown primitive type {type(obj).__name__}")

    rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
    return (
        f'<mujoco model="{obj.name}_model"><worldbody><body name="{type_str}_body" pos="0 0 0">'
        f'<geom name="{type_str}_geom" type="{type_str}" size="{size_str}" mass="{obj.mass}" rgba="{rgba_str}"/>'
        "</body></worldbody></mujoco>"
    )
