"""Material randomisation: append PBR materials from a Blender library file.

Designed for material library .blend files such as Open6DOR's
``material_lib_v2.blend`` (165 wood/rubber/metal/plastic/background
materials), but works with any .blend file as long as you point
``library_path`` at it.

The library file may be large (~1 GB+). We only ``append`` the specific
material we want to assign, on demand, and Blender caches the result so
repeated picks don't re-import. A small in-memory bookkeeping dict tracks
already-appended names so duplicates don't accumulate across many frames.

Also provides a procedural fallback (``make_random_pbr_material``) for
callers that don't have a library handy.
"""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import bpy
from loguru import logger as log

DEFAULT_CATEGORY_RE = re.compile(r"^([a-zA-Z]+)_(\d+)$")


@dataclass
class MaterialLibrary:
    """An indexed view of a .blend material library.

    Attributes:
        path: absolute path to the .blend file.
        all_names: every material name found in the library.
        by_category: dict mapping ``"wood" / "rubber" / ...`` to the matching names.
    """

    path: str
    all_names: list[str]
    by_category: dict[str, list[str]]


def index_material_library(path: str) -> MaterialLibrary:
    """Scan a .blend file's material data block names without loading them."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with bpy.data.libraries.load(path, link=False) as (data_from, _data_to):
        names = list(data_from.materials)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for n in names:
        m = DEFAULT_CATEGORY_RE.match(n)
        cat = m.group(1) if m else "misc"
        by_cat[cat].append(n)
    return MaterialLibrary(path=path, all_names=sorted(names), by_category=dict(by_cat))


def _append_material(lib_path: str, name: str) -> bpy.types.Material | None:
    """Append (copy) one material from a library file into the current .blend.

    No-op + return the existing material if it's already imported.
    """
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    with bpy.data.libraries.load(lib_path, link=False) as (data_from, data_to):
        if name not in data_from.materials:
            log.warning(f"material {name!r} not in library {lib_path}")
            return None
        data_to.materials = [name]
    return bpy.data.materials.get(name)


def _assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Replace ``obj``'s first material slot (or append one) with ``mat``."""
    if obj.type != "MESH":
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def assign_random_material(
    lib: MaterialLibrary,
    obj: bpy.types.Object,
    categories: list[str] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Pick a random material (optionally filtered by category) and assign it.

    Returns the chosen material name or None if nothing matched.
    """
    rng = rng or random
    if categories:
        pool: list[str] = []
        for c in categories:
            pool.extend(lib.by_category.get(c, []))
    else:
        pool = lib.all_names
    if not pool:
        return None
    name = rng.choice(pool)
    mat = _append_material(lib.path, name)
    if mat is None:
        return None
    _assign_material(obj, mat)
    return name


def assign_random_materials_to(
    lib: MaterialLibrary,
    objects: list[bpy.types.Object],
    categories: list[str] | None = None,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Assign a fresh random material to each object. Returns ``obj.name -> mat name``."""
    rng = rng or random
    out: dict[str, str] = {}
    for obj in objects:
        name = assign_random_material(lib, obj, categories=categories, rng=rng)
        if name is not None:
            out[obj.name] = name
    return out


def make_random_pbr_material(
    name: str,
    rng: random.Random | None = None,
) -> bpy.types.Material:
    """Procedural fallback: build a Principled-BSDF material with random params.

    Useful when no material library is available. Covers the plastic /
    metal / rubber spectrum by sweeping metallic + roughness + colour.
    """
    rng = rng or random
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    bsdf.inputs["Base Color"].default_value = (
        rng.uniform(0.05, 0.95),
        rng.uniform(0.05, 0.95),
        rng.uniform(0.05, 0.95),
        1.0,
    )
    bsdf.inputs["Metallic"].default_value = rng.choice([0.0, 0.0, 1.0])
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.15, 0.95)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = rng.uniform(0.3, 0.6)
    return mat


def assign_procedural_random_to(
    objects: list[bpy.types.Object],
    rng: random.Random | None = None,
    name_prefix: str = "RV_proc",
) -> dict[str, tuple[float, float, float]]:
    """Assign a fresh procedural PBR material to each object (no library needed)."""
    rng = rng or random
    out: dict[str, tuple[float, float, float]] = {}
    for i, obj in enumerate(objects):
        mat = make_random_pbr_material(f"{name_prefix}_{i}_{rng.randint(0, 1 << 30)}", rng=rng)
        _assign_material(obj, mat)
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            c = bsdf.inputs["Base Color"].default_value
            out[obj.name] = (c[0], c[1], c[2])
    return out
