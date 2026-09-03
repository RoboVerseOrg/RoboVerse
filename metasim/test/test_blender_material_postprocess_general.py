from __future__ import annotations

import ast
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_PATH = REPO_ROOT / "metasim" / "sim" / "blender" / "blender.py"

_HELPER_NAMES = {
    "_apply_usd_material_specs",
    "_custom_property_value",
    "_repair_imported_materials",
    "_roboverse_custom_property_owners",
    "has_preserved_roboverse_material_graph",
}
_CONSTANT_NAMES = {
    "_PRESERVED_ROBOVERSE_MATERIAL_POLICIES",
    "_ROBOVERSE_MATERIAL_GRAPH_VERSION",
    "_ROBOVERSE_MATERIAL_GRAPH_VERSION_KEY",
    "_ROBOVERSE_MATERIAL_POLICY_KEY",
}


def _target_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _load_blender_material_helpers() -> types.ModuleType:
    source = BLENDER_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(BLENDER_PATH))
    selected: list[ast.stmt] = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _target_names(node) & _CONSTANT_NAMES:
            selected.append(node)

    module_ast = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    helpers = types.ModuleType("blender_material_helpers")
    helpers.__dict__["UsdMaterialSpec"] = object
    exec(compile(module_ast, str(BLENDER_PATH), "exec"), helpers.__dict__)
    return helpers


class FakeNode(dict):
    pass


class FakeNodeTree(dict):
    def __init__(self, nodes: list[FakeNode] | None = None, custom_properties: dict[str, object] | None = None):
        super().__init__(custom_properties or {})
        self.nodes = nodes or []


class FakeMaterial(dict):
    def __init__(
        self,
        name: str,
        custom_properties: dict[str, object] | None = None,
        node_tree: FakeNodeTree | None = None,
    ):
        super().__init__(custom_properties or {})
        self.name = name
        self.node_tree = node_tree


class FakeSlot:
    def __init__(self, material: FakeMaterial | None):
        self.material = material


class FakeObject:
    def __init__(
        self,
        name: str,
        materials: list[FakeMaterial],
        children: tuple[FakeObject, ...] = (),
    ):
        self.name = name
        self.type = "MESH"
        self.material_slots = [FakeSlot(material) for material in materials]
        self.children = children
        self.instance_collection = None


def _tagged_material(name: str, policy: str, version: str = "v1") -> FakeMaterial:
    return FakeMaterial(
        name,
        {
            "roboverse:material_conversion_policy": policy,
            "roboverse:material_graph_version": version,
        },
    )


def test_has_preserved_roboverse_material_graph_detects_material_and_node_tags() -> None:
    helpers = _load_blender_material_helpers()
    node_tagged = FakeMaterial(
        "NodeTagged",
        node_tree=FakeNodeTree(
            nodes=[
                FakeNode({
                    "roboverse:material_conversion_policy": "mdl_bake",
                    "roboverse:material_graph_version": "v1",
                })
            ]
        ),
    )

    assert helpers.has_preserved_roboverse_material_graph(_tagged_material("Direct", "direct_graph"))
    assert helpers.has_preserved_roboverse_material_graph(node_tagged)

    for policy in ("class_fallback", "failed", "missing", "broken"):
        assert not helpers.has_preserved_roboverse_material_graph(_tagged_material(policy, policy))
    assert not helpers.has_preserved_roboverse_material_graph(_tagged_material("WrongVersion", "direct_graph", "v0"))
    assert not helpers.has_preserved_roboverse_material_graph(FakeMaterial("MissingTags"))


def test_repair_imported_materials_skips_only_preserved_roboverse_graphs() -> None:
    helpers = _load_blender_material_helpers()
    repaired: list[tuple[str, tuple[float, float, float, float] | None]] = []
    helpers._material_name_candidates = lambda name: (name,)
    helpers._repair_material_surface_shader = lambda material, rgba=None: repaired.append((material.name, rgba))
    preserved = _tagged_material("Preserved", "direct_graph")
    class_fallback = _tagged_material("ClassFallback", "class_fallback")
    failed = _tagged_material("Failed", "failed")

    helpers._repair_imported_materials(
        [FakeObject("Imported", [preserved, class_fallback, failed])],
        {
            "ClassFallback": (0.1, 0.2, 0.3, 1.0),
            "Failed": (0.4, 0.5, 0.6, 1.0),
            "Preserved": (0.7, 0.8, 0.9, 1.0),
        },
    )

    assert repaired == [
        ("ClassFallback", (0.1, 0.2, 0.3, 1.0)),
        ("Failed", (0.4, 0.5, 0.6, 1.0)),
    ]


def test_apply_usd_material_specs_skips_preserved_roboverse_graphs() -> None:
    helpers = _load_blender_material_helpers()
    applied: list[tuple[str, object]] = []
    helpers._material_name_candidates = lambda name: (name,)
    helpers._apply_usd_material_spec_to_material = lambda material, spec: applied.append((material.name, spec))
    preserved = _tagged_material("Preserved", "mdl_bake")
    repairable = _tagged_material("Repairable", "class_fallback")

    helpers._apply_usd_material_specs(
        [FakeObject("Imported", [preserved, repairable])],
        {
            "Preserved": object(),
            "Repairable": "spec-repairable",
        },
    )

    assert applied == [("Repairable", "spec-repairable")]
