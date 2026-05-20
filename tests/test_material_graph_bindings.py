from pathlib import Path

import pytest

from roboverse_pack.blender.usd.material_graph.adapters.omnipbr import _common_uv_primvars
from roboverse_pack.blender.usd.material_graph.bindings import (
    _uv_primvars,
    collect_material_binding_contexts,
    resolve_uv_set,
)
from roboverse_pack.blender.usd.material_graph.context import MaterialContext


class _Attr:
    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name


class _Prim:
    def __init__(self, attr_names, path="/World/Prim", is_gprim=True):
        self._attr_names = attr_names
        self._path = path
        self._is_gprim = is_gprim

    def GetAttributes(self):
        return [_Attr(name) for name in self._attr_names]

    def GetPath(self):
        return self._path

    def IsA(self, schema):
        return schema is _UsdGeom.Gprim and self._is_gprim


class _Material:
    def GetPath(self):
        return "/World/Looks/Mat"


class _Binding:
    def __init__(self, prim):
        self._prim = prim

    def ComputeBoundMaterial(self):
        return (_Material(), None)


class _UsdShade:
    @staticmethod
    def MaterialBindingAPI(prim):
        return _Binding(prim)


class _UsdGeom:
    class Gprim:
        pass

    class Mesh:
        pass


class _Stage:
    def __init__(self, prims):
        self._prims = prims

    def Traverse(self):
        return tuple(self._prims)


def _context(uv_primvars_by_prim=None):
    return MaterialContext(
        source_path=Path("scene.usda"),
        texture_base_dir=Path("."),
        material_path="/World/Looks/Mat",
        bound_prim_paths=tuple(uv_primvars_by_prim or ()),
        uv_primvars_by_prim=uv_primvars_by_prim,
    )


@pytest.mark.parametrize(
    ("requested", "available", "expected_name", "expected_status"),
    [
        (None, ("uv", "st"), "st", "matched"),
        (0, ("uv0", "map1"), "uv0", "matched"),
        (1, ("map2", "uv2", "st1"), "st1", "matched"),
        (2, ("uv3", "map3"), "uv3", "matched"),
        (3, ("st", "uv"), "st", "guessed_or_missing"),
    ],
)
def test_resolve_uv_set_preference_order(requested, available, expected_name, expected_status):
    spec = resolve_uv_set(requested, available)

    assert spec.primvar_name == expected_name
    assert spec.requested_index == requested
    assert spec.resolution_status == expected_status


def test_uv_primvars_extracts_valid_primvar_names_from_fake_prim():
    prim = _Prim(
        [
            "primvars:st",
            "primvars:st1",
            "primvars:uv",
            "primvars:uv2",
            "primvars:map1",
            "primvars:displayColor",
            "points",
        ]
    )

    assert _uv_primvars(prim, UsdGeom=None) == ("map1", "st", "st1", "uv", "uv2")


def test_common_uv_primvars_returns_sorted_intersection_for_shared_sets():
    context = _context(
        {
            "/World/MeshA": ("st", "st1", "uv"),
            "/World/MeshB": ("st1", "st", "map1"),
        }
    )

    assert _common_uv_primvars(context) == ("st", "st1")


def test_common_uv_primvars_defaults_to_st_for_disjoint_sets():
    context = _context(
        {
            "/World/MeshA": ("st1",),
            "/World/MeshB": ("uv",),
        }
    )

    assert _common_uv_primvars(context) == ("st",)


def test_common_uv_primvars_defaults_to_st_without_bindings():
    assert _common_uv_primvars(_context()) == ("st",)


def test_collect_material_binding_contexts_skips_bound_container_prims_without_uvs():
    container = _Prim([], path="/World/Container", is_gprim=False)
    mesh = _Prim(["primvars:st1"], path="/World/Container/Mesh", is_gprim=True)

    contexts = collect_material_binding_contexts(_Stage([container, mesh]), _UsdShade, _UsdGeom)

    context = contexts["/World/Looks/Mat"]
    assert context.bound_prim_paths == ("/World/Container/Mesh",)
    assert context.uv_primvars_by_prim == {"/World/Container/Mesh": ("st1",)}
    assert _common_uv_primvars(
        MaterialContext(
            source_path=Path("scene.usda"),
            texture_base_dir=Path("."),
            material_path="/World/Looks/Mat",
            bound_prim_paths=context.bound_prim_paths,
            uv_primvars_by_prim=context.uv_primvars_by_prim,
        )
    ) == ("st1",)


def test_collect_material_binding_contexts_with_pxr_stage():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from roboverse_pack.blender.usd.material_graph.bindings import collect_material_binding_contexts

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    mesh.GetPrim().CreateAttribute("primvars:st1", Sdf.ValueTypeNames.TexCoord2fArray)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    contexts = collect_material_binding_contexts(stage, UsdShade, UsdGeom)

    assert contexts["/World/Looks/Mat"].material_path == "/World/Looks/Mat"
    assert contexts["/World/Looks/Mat"].bound_prim_paths == ("/World/Mesh",)
    assert contexts["/World/Looks/Mat"].uv_primvars_by_prim == {"/World/Mesh": ("st1",)}
