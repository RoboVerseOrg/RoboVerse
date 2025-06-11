from __future__ import annotations

import os
import random
from typing import Iterator, Literal

import numpy as np
import omni
import yaml
from loguru import logger as log
from omni.kit.material.library import get_material_prim_path
from pxr import Gf, Sdf, Usd, UsdShade

from .material_util import apply_mdl_to_prim

try:
    import omni.isaac.core.utils.prims as prim_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.prims as prim_utils


def prim_descendants(root_prim: Usd.Prim) -> Iterator[Usd.Prim]:
    for prim in root_prim.GetAllChildren():
        yield prim
        yield from prim_descendants(prim)


class ShaderFixer:
    def __init__(self, usd_path: str, root_prim_path: str):
        stage = omni.usd.get_context().get_stage()
        self.usd_path = usd_path
        self.root_prim = stage.GetPrimAtPath(root_prim_path)
        self.pairs: list[tuple[Usd.Prim, UsdShade.Material, UsdShade.Shader]] = []  # obj_prim, material, shader
        self._find_all_targets()

    def _find_all_targets(self):
        for prim in prim_descendants(self.root_prim):
            material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
            if material:
                shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
                if shader:
                    self.pairs.append((prim, material, shader))
                else:
                    log.warning(f"No shader found for material {material.GetPrim().GetPath()}")

    def _find_unique_shaders(self) -> list[UsdShade.Shader]:
        unique_shaders = {}
        for _, _, shader in self.pairs:
            shader_path = shader.GetPrim().GetPath()
            if shader_path not in unique_shaders:
                unique_shaders[shader_path] = shader
        return list(unique_shaders.values())

    def _fix_single_shader(self, shader: UsdShade.Shader):
        shader_prim = shader.GetPrim()  # often == material_prim + '/Shader'
        log.debug(f"Trying to fix shader {shader_prim.GetPath()}")

        target_attrs = ["diffuse_texture", "opacity_texture"]
        for attr_name in target_attrs:
            attr = shader_prim.GetAttribute(f"inputs:{attr_name}")
            if not attr:
                log.debug(f"No {attr_name} attribute found for shader, skipping")
                continue

            original_path = str(attr.Get()).replace("@", "")
            if os.path.isabs(original_path):
                if os.path.exists(original_path):
                    log.error(f"{attr_name} path {original_path} is absolute, please change to relative path!")
                else:
                    log.error(
                        f"{attr_name} path {original_path} is absolute and does not exist, please fix manually as"
                        " relative path!"
                    )
            else:
                usd_abs_path = os.path.abspath(self.usd_path)
                abs_path = os.path.join(os.path.dirname(usd_abs_path), original_path)
                if os.path.exists(abs_path):
                    attr.Set(Sdf.AssetPath(abs_path))
                    log.info(
                        f"Fixing shader {shader_prim.GetPath()}, replace {attr_name} path from {original_path} to"
                        f" {abs_path}"
                    )
                else:
                    log.error(
                        f"{attr_name} path {abs_path} (parsed from {original_path}) does not exist! If you are developer, please fix the path. If you are user, please report this issue."
                    )

    def fix_all(self):
        for shader in self._find_unique_shaders():
            self._fix_single_shader(shader)


class ReflectionRandomizer:
    def __init__(self, root_prim_path: str = "/World/envs", change_existing_color: bool = True):
        self.root_prim = prim_utils.get_prim_at_path(root_prim_path)
        self.stage = omni.usd.get_context().get_stage()
        self.change_existing_color = change_existing_color

    def _single(self, prim: Usd.Prim):
        material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        if not material:
            # log.error(f"No material found for {prim.GetPath()}")
            mtl_name = f"material_{random.randint(0, 1000000)}"
            _, mtl_prim_path = get_material_prim_path(mtl_name)
            material = UsdShade.Material.Define(self.stage, mtl_prim_path)
            omni.kit.commands.execute("BindMaterial", prim_path=prim.GetPath(), material_path=mtl_prim_path)

        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
        if not shader:
            # log.error(f"No shader found for {prim.GetPath()}")
            # see https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/materials/create-usdpreviewsurface-material.html
            shader = UsdShade.Shader.Define(self.stage, mtl_prim_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            if self.change_existing_color:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*np.random.uniform(0.0, 1.0, 3).tolist())
                )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        if shader.GetImplementationSource() == UsdShade.Tokens.sourceAsset:
            ## already has mdl
            for k in ["reflection_roughness_constant", "metallic_constant", "specular_level"]:
                shader.CreateInput(k, Sdf.ValueTypeNames.Float).Set(random.uniform(0.0, 1.0))
        elif shader.GetImplementationSource() == UsdShade.Tokens.id:
            ## UsdPreviewSurface (primitive objects use this)
            for k in ["roughness", "metallic"]:
                shader.CreateInput(k, Sdf.ValueTypeNames.Float).Set(random.uniform(0.0, 1.0))
            if self.change_existing_color:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*np.random.uniform(0.0, 1.0, 3).tolist())
                )
        else:
            log.warning(f"Unknown shader implementation source: {shader.GetImplementationSource()}")

    def do_it(self):
        for prim in prim_descendants(self.root_prim):
            self._single(prim)


def _get_mdl_paths_from_yaml(
    yaml_path: str,
    split: Literal["train", "test", "val", "all"] = "all",
    datasets: list[Literal["arnold", "vMaterials_2"]] = ("arnold", "vMaterials_2"),
) -> list[str]:
    splits = ["train", "test", "val"] if split == "all" else [split]
    data = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    mdl_paths = []
    for split in splits:
        for dataset in datasets:
            mdl_paths.extend(data[dataset][split])
    mdl_paths = list(set(mdl_paths))
    return mdl_paths


class MdlRandomizer:
    def __init__(
        self,
        prim_path: str,
        case: Literal["table", "ground", "wall"],
        split: Literal["train", "val", "test", "all"] = "all",
        datasets: list[Literal["arnold", "vMaterials_2"]] = ("arnold", "vMaterials_2"),
    ):
        """
        Args:
            prim_path: the path of the prim to be randomized
            case: the type of the prim to be randomized, can be "table", "ground", or "wall"
            split: the split of the mdl files to be used, can be "train", "val", "test", "all"
            datasets: the datasets to be used
        """
        self.prim_path = prim_path
        self.case = case
        self.split = split
        self.datasets = datasets

        self.candicates = _get_mdl_paths_from_yaml(
            f"metasim/sim/isaaclab/cfg/randomization/{case}_mdl_paths.yml", split, datasets
        )
        log.info(
            f"Initializing {case} randomizer with datasets: {datasets}, split: {split},"
            f" total candidates: {len(self.candicates)}"
        )

    def do_it(self):
        mdl_path = random.choice(self.candicates)
        log.info(f"Randomizing albedo map for {self.case} with {mdl_path}")
        apply_mdl_to_prim(mdl_path, self.prim_path)


class SceneRandomizer:
    def __init__(self):
        import metasim.cfg.scenes as scenes_module
        from metasim.cfg.scenes import SceneCfg

        self.scene_configs = []
        for attr_name in dir(scenes_module):
            attr = getattr(scenes_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, SceneCfg) and attr != SceneCfg:
                self.scene_configs.append(attr)

        log.info(f"Initialized SceneRandomizer with {len(self.scene_configs)} scenes")

    def do_it(self):
        scene_cfg_cls = random.choice(self.scene_configs)
        scene_cfg = scene_cfg_cls()
        # check_and_download_recursive([scene_cfg.usd_path])  # XXX: this could be buggy
        chosen_position = random.choice(scene_cfg.positions) if scene_cfg.positions else scene_cfg.default_position

        return {
            "filepath": scene_cfg.usd_path,
            "position": chosen_position,
            "quat": scene_cfg.quat,
            "scale": scene_cfg.scale,
        }
