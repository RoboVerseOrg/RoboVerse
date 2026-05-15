from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
MJCF_SUFFIXES = {".xml", ".mjcf"}
USD_LIGHT_ENERGY_SCALE = 1.75e-5


@dataclass(frozen=True)
class BlenderImportResult:
    root: object
    imported: list[object]
    body_objects: dict[str, object]


def _as_float_tuple(values, *, length: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"Expected {length} values, got {len(result)}: {values!r}")
    return result


def _usd_import_kwargs() -> dict[str, object]:
    """Use authored USD data instead of Blender's sparse defaults when importing stages."""
    props = bpy.ops.wm.usd_import.get_rna_type().properties
    desired = {
        "import_materials": True,
        "import_all_materials": True,
        "import_usd_preview": True,
        "import_lights": True,
        "create_world_material": True,
        "mtl_purpose": "MTL_FULL",
        "property_import_mode": "ALL",
    }
    return {name: value for name, value in desired.items() if name in props}


def _scale_imported_usd_light_energy(imported: list[object], scale: float = USD_LIGHT_ENERGY_SCALE) -> None:
    """Map authored USD light intensities into Blender Cycles' energy range."""
    for obj in imported:
        if getattr(obj, "type", None) != "LIGHT":
            continue
        data = getattr(obj, "data", None)
        if data is None or not hasattr(data, "energy"):
            continue
        data.energy = float(data.energy) * float(scale)


def _scene_collections() -> set[object]:
    collections = {bpy.context.scene.collection}
    pending = list(bpy.context.scene.collection.children)
    while pending:
        collection = pending.pop()
        if collection in collections:
            continue
        collections.add(collection)
        pending.extend(collection.children)
    return collections


def _collection_parent_map() -> dict[object, object]:
    parents: dict[object, object] = {}

    def visit(parent) -> None:
        for child in parent.children:
            parents[child] = parent
            visit(child)

    visit(bpy.context.scene.collection)
    return parents


def _imported_prototype_collections(imported: list[object]) -> set[object]:
    return {
        obj.instance_collection
        for obj in imported
        if getattr(obj, "instance_collection", None) is not None
    }


def _prototype_collection_closure(prototype_collections: set[object]) -> set[object]:
    parents = _collection_parent_map()
    closure = set(prototype_collections)
    changed = True
    while changed:
        changed = False
        for child, parent in tuple(parents.items()):
            if child not in closure or parent in closure or parent == bpy.context.scene.collection:
                continue
            if len(parent.objects) > 0:
                continue
            if all(grandchild in closure for grandchild in parent.children):
                closure.add(parent)
                changed = True
    return closure


def _detach_imported_prototype_collections(imported: list[object]) -> set[object]:
    """Keep Blender USD prototype collections available only through collection instances."""
    prototype_collections = _prototype_collection_closure(_imported_prototype_collections(imported))
    if not prototype_collections:
        return set()

    parents = _collection_parent_map()
    for collection in sorted(prototype_collections, key=lambda item: item.name):
        parent = parents.get(collection)
        if parent is None or parent in prototype_collections:
            continue
        parent.children.unlink(collection)
    return prototype_collections


def _parent_imported_scene_roots(imported: list[object], root: object) -> None:
    scene_collections = _scene_collections()
    prototype_collections = _imported_prototype_collections(imported)
    for obj in imported:
        if obj.parent is not None:
            continue
        if any(collection in prototype_collections for collection in obj.users_collection):
            continue
        if not any(collection in scene_collections for collection in obj.users_collection):
            continue
        obj.parent = root


def import_usd_visuals(
    filepath: str | Path,
    *,
    root_name: str,
    default_position=(0.0, 0.0, 0.0),
    default_orientation=(1.0, 0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
) -> BlenderImportResult:
    path = Path(filepath)
    if path.suffix.lower() not in USD_SUFFIXES:
        raise ValueError(f"{root_name!r} expected USD suffix {sorted(USD_SUFFIXES)}, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"USD asset for {root_name!r} does not exist: {path}")

    before_names = set(bpy.data.objects.keys())
    result = bpy.ops.wm.usd_import(filepath=str(path), **_usd_import_kwargs())
    if "FINISHED" not in result:
        raise RuntimeError(f"Failed to import USD for {root_name!r}: {result}")

    imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
    if not imported:
        raise RuntimeError(f"USD import for {root_name!r} produced no Blender objects: {path}")
    _scale_imported_usd_light_energy(imported)
    _detach_imported_prototype_collections(imported)

    root = bpy.data.objects.new(root_name, None)
    bpy.context.collection.objects.link(root)
    root.location = _as_float_tuple(default_position, length=3)
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = _as_float_tuple(default_orientation, length=4)
    root.scale = _as_float_tuple(scale, length=3)

    _parent_imported_scene_roots(imported, root)

    return BlenderImportResult(root=root, imported=imported, body_objects={})
