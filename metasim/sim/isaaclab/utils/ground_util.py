import omni
from pxr import Sdf, UsdShade

# from .material_util import apply_mdl_to_prim

try:
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.lab.sim as sim_utils
except ModuleNotFoundError:
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils

GROUND_PRIM_PATH = "/World/ground"


def create_ground():
    # original file is http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png
    cfg_ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        color=(1.0, 1.0, 1.0),
    )
    cfg_ground.func(GROUND_PRIM_PATH, cfg_ground)


def set_ground_material_scale(scale: float):
    return
    ground_prim = prim_utils.get_prim_at_path(GROUND_PRIM_PATH)
    material = UsdShade.MaterialBindingAPI(ground_prim).GetDirectBinding().GetMaterial()
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
    # Correspond to Shader -> Inputs -> UV -> Texture Tiling (in Isaac Sim 4.2.0)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set((scale, scale))


def set_ground_material(material_mdl_path: str):
    pass
    # apply_mdl_to_prim(material_mdl_path, GROUND_PRIM_PATH)
