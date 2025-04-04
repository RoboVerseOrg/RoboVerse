import os

import omni
from omni.kit.material.library import get_material_prim_path
from pxr import UsdShade

try:
    import omni.isaac.core.utils.prims as prim_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.prims as prim_utils


def apply_mdl_to_prim(mdl_path: str, prim_path: str):
    assert os.path.exists(mdl_path), f"Material file {mdl_path} does not exist"
    assert mdl_path.endswith(".mdl"), f"Material file {mdl_path} must have .mdl extension"
    prim = prim_utils.get_prim_at_path(prim_path)

    mtl_name = os.path.basename(mdl_path).removesuffix(".mdl")
    _, mtl_prim_path = get_material_prim_path(mtl_name)
    success, result = omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url=mdl_path,
        mtl_name=mtl_name,
        mtl_path=mtl_prim_path,
        select_new_prim=False,
    )
    if not success:
        raise RuntimeError(f"Failed to create material {mtl_name} at {mtl_prim_path}")

    success, result = omni.kit.commands.execute(
        "BindMaterial",
        prim_path=prim.GetPath(),
        material_path=mtl_prim_path,
        strength=UsdShade.Tokens.strongerThanDescendants,
    )
    if not success:
        raise RuntimeError(f"Failed to bind material at {mtl_prim_path} to {prim.GetPath()}")
