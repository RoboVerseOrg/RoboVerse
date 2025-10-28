from __future__ import annotations

import dataclasses
import os
import random
from typing import Any, Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass
from metasim.utils.hf_util import check_and_download_single


def extract_texture_paths_from_mdl(mdl_file_path: str) -> list[str]:
    """Extract texture file paths referenced in an MDL file by parsing its content.

    Args:
        mdl_file_path: Path to the MDL file

    Returns:
        List of absolute texture file paths referenced in the MDL file
    """
    texture_paths = []

    if not os.path.exists(mdl_file_path):
        return texture_paths

    mdl_dir = os.path.dirname(mdl_file_path)

    try:
        with open(mdl_file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse texture_2d declarations in MDL files
        # Pattern: texture_2d("./path/to/texture.png", optional_args)
        import re

        texture_pattern = r'texture_2d\("([^"]+)"[^)]*\)'
        matches = re.findall(texture_pattern, content)

        for match in matches:
            if match.strip():  # Skip empty texture declarations
                # Convert relative paths to absolute paths
                if match.startswith("./"):
                    texture_path = os.path.join(mdl_dir, match[2:])  # Remove './'
                elif match.startswith("../"):
                    texture_path = os.path.abspath(os.path.join(mdl_dir, match))
                elif not os.path.isabs(match):
                    texture_path = os.path.join(mdl_dir, match)
                else:
                    texture_path = match

                texture_paths.append(os.path.normpath(texture_path))

    except Exception as e:
        logger.warning(f"Failed to parse MDL file {mdl_file_path}: {e}")

    return texture_paths


def extract_material_name_from_mdl(mdl_file_path: str) -> str | None:
    """Extract the actual material name from an MDL file.

    Args:
        mdl_file_path: Path to the MDL file

    Returns:
        The material name found in the MDL file, or None if not found
    """
    if not os.path.exists(mdl_file_path):
        return None

    try:
        with open(mdl_file_path, encoding="utf-8") as f:
            content = f.read()

        import re

        material_pattern = r"(?:export\s+)?material\s+(\w+)\s*\("  # Cork_mat --> Cork_mat
        match = re.search(material_pattern, content)

        if match:
            return match.group(1)

    except Exception as e:
        logger.warning(f"Failed to extract material name from MDL file {mdl_file_path}: {e}")

    return None


@configclass
class PhysicalMaterialCfg:
    """Configuration for physical material properties.

    Args:
        friction_range: Range for friction coefficient randomization
        restitution_range: Range for restitution (bounciness) randomization
        distribution: Type of distribution for random sampling
        enabled: Whether to apply physical material randomization
    """

    friction_range: tuple[float, float] | None = None
    restitution_range: tuple[float, float] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class PBRMaterialCfg:
    """Configuration for PBR (Physically Based Rendering) material properties.

    Args:
        roughness_range: Range for surface roughness (0=smooth, 1=rough)
        metallic_range: Range for metallic property (0=dielectric, 1=metallic)
        specular_range: Range for specular reflection intensity
        diffuse_color_range: RGB color ranges as ((r_min,r_max), (g_min,g_max), (b_min,b_max))
        distribution: Type of distribution for random sampling
        enabled: Whether to apply PBR randomization
    """

    roughness_range: tuple[float, float] | None = None
    metallic_range: tuple[float, float] | None = None
    specular_range: tuple[float, float] | None = None
    diffuse_color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class MDLMaterialCfg:
    """Configuration for MDL (Material Definition Language) files.

    Args:
        mdl_paths: List of paths to MDL material files (can be empty, will be populated by presets)
        selection_strategy: How to select from available MDL files
        enabled: Whether to apply MDL material randomization
        auto_download: Whether to automatically download missing MDL files
        validate_paths: Whether to validate file existence at initialization
    """

    mdl_paths: list[str] = dataclasses.field(default_factory=list)
    selection_strategy: Literal["random", "sequential", "weighted"] = "random"
    weights: list[float] | None = None
    enabled: bool = True
    auto_download: bool = True
    validate_paths: bool = True

    def __post_init__(self):
        """Validate MDL configuration."""
        # Allow empty mdl_paths - will be handled by presets or user configuration
        if self.enabled and not self.mdl_paths:
            logger.warning(
                "MDL material randomization is enabled but no paths provided. Use material presets or provide paths manually."
            )

        if self.mdl_paths and self.selection_strategy == "weighted":
            if self.weights is None or len(self.weights) != len(self.mdl_paths):
                raise ValueError("weights must be provided and match mdl_paths length for weighted selection")


@configclass
class MaterialRandomCfg:
    """Unified configuration for material randomization.

    Args:
        obj_name: Name of the object to randomize materials for
        physical: Physical material properties configuration (optional)
        pbr: PBR material properties configuration (optional)
        mdl: MDL material files configuration (optional)
        env_ids: List of environment IDs to apply randomization to (None = all)
        randomization_mode: How to apply multiple material types
            - 'combined': Apply physics + best available visual (MDL > PBR)
            - 'physics_only': Apply only physical properties
            - 'visual_only': Apply only visual properties (MDL > PBR)
            - 'all_separate': Apply all enabled types independently
    """

    obj_name: str = dataclasses.MISSING
    physical: PhysicalMaterialCfg | None = None
    pbr: PBRMaterialCfg | None = None
    mdl: MDLMaterialCfg | None = None
    env_ids: list[int] | None = None
    randomization_mode: Literal["combined", "physics_only", "visual_only", "all_separate"] = "combined"

    def __post_init__(self):
        """Validate configuration."""
        available_configs = [cfg for cfg in [self.physical, self.pbr, self.mdl] if cfg is not None]
        if not available_configs:
            # If no configurations provided, create a default PBR configuration
            logger.warning(
                f"No material configurations provided for {self.obj_name}. Creating default PBR configuration."
            )
            self.pbr = PBRMaterialCfg(
                roughness_range=(0.0, 1.0),
                metallic_range=(0.0, 1.0),
                diffuse_color_range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                enabled=True,
            )
            available_configs = [self.pbr]

        enabled_configs = [cfg for cfg in available_configs if getattr(cfg, "enabled", True)]
        if not enabled_configs:
            raise ValueError("At least one material type must be enabled")


class MaterialRandomizer(BaseRandomizerType):
    """Material randomizer supporting Physical, PBR, and MDL materials.

    Supports multiple randomization modes and distributions with reproducible seeding.
    """

    def __init__(self, cfg: MaterialRandomCfg, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self._mdl_selection_state = {"sequential_index": 0}

        # Set up reproducible random state - simple and direct
        if seed is not None:
            # Use provided seed + simple string-to-number conversion for uniqueness
            name_sum = sum(ord(c) for c in cfg.obj_name)
            self._seed = seed + name_sum
        else:
            self._seed = random.randint(0, 2**32 - 1)

        self._rng = random.Random(self._seed)
        logger.debug(f"MaterialRandomizer for '{cfg.obj_name}' using seed {self._seed}")

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Bind the handler to the randomizer."""
        mod = handler.__class__.__module__

        if mod.startswith("metasim.sim.isaacsim"):
            super().bind_handler(handler, *args, **kwargs)
            # Import IsaacSim specific modules only when needed
            try:
                global omni, prim_utils, get_material_prim_path, Gf, Sdf, UsdShade
                import omni

                # Use the same import pattern as the project's material_util.py
                try:
                    import omni.isaac.core.utils.prims as prim_utils
                except ModuleNotFoundError:
                    import isaacsim.core.utils.prims as prim_utils

                from omni.kit.material.library import get_material_prim_path
                from pxr import Gf, Sdf, UsdShade

                self.stage = omni.usd.get_context().get_stage()
            except ImportError as e:
                raise ImportError(f"Failed to import IsaacSim modules: {e}") from e
        else:
            raise ValueError(f"Unsupported handler type: {type(handler)} for MaterialRandomizer")

    def _get_object_instance(self, obj_name: str):
        """Get object instance from handler."""
        if obj_name in self.handler.scene.articulations:
            return self.handler.scene.articulations[obj_name]
        elif obj_name in self.handler.scene.rigid_objects:
            return self.handler.scene.rigid_objects[obj_name]
        else:
            raise ValueError(f"Object {obj_name} not found in scene")

    def _get_env_ids(self) -> list[int]:
        """Get environment IDs to operate on."""
        return self.cfg.env_ids or list(range(self.handler.num_envs))

    def _generate_random_value(self, value_range: tuple[float, float], distribution: str = "uniform") -> float:
        """Generate a single random value using reproducible RNG."""
        if distribution == "uniform":
            return self._rng.uniform(value_range[0], value_range[1])
        elif distribution == "log_uniform":
            log_min = torch.log(torch.tensor(value_range[0])).item()
            log_max = torch.log(torch.tensor(value_range[1])).item()
            return torch.exp(torch.tensor(self._rng.uniform(log_min, log_max))).item()
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            val = self._rng.gauss(mean, std)
            return max(value_range[0], min(value_range[1], val))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _generate_random_tensor(
        self, value_range: tuple[float, float], shape: tuple, distribution: str = "uniform"
    ) -> torch.Tensor:
        """Generate random tensor with specified shape."""
        if distribution == "uniform":
            return torch.rand(shape) * (value_range[1] - value_range[0]) + value_range[0]
        elif distribution == "log_uniform":
            log_min = torch.log(torch.tensor(value_range[0]))
            log_max = torch.log(torch.tensor(value_range[1]))
            return torch.exp(torch.rand(shape) * (log_max - log_min) + log_min)
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            values = torch.normal(mean, std, size=shape)
            return torch.clamp(values, value_range[0], value_range[1])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def randomize_physical_properties(self) -> None:
        """Randomize physical material properties (friction, restitution)."""
        if not self.cfg.physical or not self.cfg.physical.enabled:
            return

        obj_inst = self._get_object_instance(self.cfg.obj_name)
        env_ids = self._get_env_ids()

        try:
            # Get current material properties
            materials = obj_inst.root_physx_view.get_material_properties()

            # Randomize friction
            if self.cfg.physical.friction_range:
                friction_vals = self._generate_random_tensor(
                    self.cfg.physical.friction_range, (len(env_ids), materials.shape[1]), self.cfg.physical.distribution
                )
                materials[env_ids, :, 0] = friction_vals  # Static friction
                materials[env_ids, :, 1] = friction_vals  # Dynamic friction

            # Randomize restitution
            if self.cfg.physical.restitution_range:
                restitution_vals = self._generate_random_tensor(
                    self.cfg.physical.restitution_range,
                    (len(env_ids), materials.shape[1]),
                    self.cfg.physical.distribution,
                )
                materials[env_ids, :, 2] = restitution_vals

            obj_inst.root_physx_view.set_material_properties(materials, torch.tensor(env_ids))

        except Exception as e:
            logger.warning(f"Failed to randomize physical properties for {self.cfg.obj_name}: {e}")

    def randomize_pbr_properties(self) -> None:
        """Randomize PBR material properties."""
        if not self.cfg.pbr or not self.cfg.pbr.enabled:
            return

        obj_inst = self._get_object_instance(self.cfg.obj_name)
        env_ids = self._get_env_ids()

        # Get prim paths for each environment
        root_path = obj_inst.cfg.prim_path

        for env_id in env_ids:
            env_prim_path = root_path.replace("env_.*", f"env_{env_id}")
            try:
                self._randomize_prim_pbr(env_prim_path)
            except Exception as e:
                logger.warning(f"Failed to randomize PBR for {env_prim_path}: {e}")

    def _randomize_prim_pbr(self, prim_path: str) -> None:
        """Randomize PBR properties for a specific prim."""
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim:
            return

        # Always create a new material to ensure it overrides any existing material
        mtl_name = f"pbr_material_{self._rng.randint(0, 1000000)}"
        _, mtl_prim_path = get_material_prim_path(mtl_name)
        material = UsdShade.Material.Define(self.stage, mtl_prim_path)

        # Bind with stronger priority to override existing materials
        success = omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim.GetPath(),
            material_path=mtl_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        if not success:
            logger.warning(f"Failed to bind PBR material to {prim.GetPath()}")
            return

        # Get or create shader
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
        if not shader:
            shader_path = material.GetPrim().GetPath().AppendChild("Shader")
            shader = UsdShade.Shader.Define(self.stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Randomize properties
        if self.cfg.pbr.roughness_range:
            val = self._generate_random_value(self.cfg.pbr.roughness_range, self.cfg.pbr.distribution)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(val)

        if self.cfg.pbr.metallic_range:
            val = self._generate_random_value(self.cfg.pbr.metallic_range, self.cfg.pbr.distribution)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(val)

        if self.cfg.pbr.specular_range:
            val = self._generate_random_value(self.cfg.pbr.specular_range, self.cfg.pbr.distribution)
            shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(val)

        if self.cfg.pbr.diffuse_color_range:
            r = self._generate_random_value(self.cfg.pbr.diffuse_color_range[0], self.cfg.pbr.distribution)
            g = self._generate_random_value(self.cfg.pbr.diffuse_color_range[1], self.cfg.pbr.distribution)
            b = self._generate_random_value(self.cfg.pbr.diffuse_color_range[2], self.cfg.pbr.distribution)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))

    def randomize_mdl_material(self) -> None:
        """Apply MDL material based on selection strategy."""
        if not self.cfg.mdl or not self.cfg.mdl.enabled:
            return

        obj_inst = self._get_object_instance(self.cfg.obj_name)
        env_ids = self._get_env_ids()

        # Select MDL path from all configured paths (regardless of existence)
        if not self.cfg.mdl.mdl_paths:
            logger.warning(f"No MDL paths configured for {self.cfg.obj_name}")
            return

        root_path = obj_inst.cfg.prim_path

        for env_id in env_ids:
            env_prim_path = root_path.replace("env_.*", f"env_{env_id}")

            # Select MDL path from all configured paths
            mdl_path = self._select_mdl_path(self.cfg.mdl.mdl_paths)

            # Download the selected MDL file if it doesn't exist
            if not os.path.exists(mdl_path) and self.cfg.mdl.auto_download:
                try:
                    logger.info(f"Downloading MDL file: {os.path.basename(mdl_path)}")
                    check_and_download_single(mdl_path)
                except Exception as e:
                    logger.warning(f"Failed to download MDL {mdl_path}: {e}")
                    continue  # Skip this environment if download fails

            # Download textures for the selected MDL file
            if os.path.exists(mdl_path):
                try:
                    texture_paths = extract_texture_paths_from_mdl(mdl_path)
                    missing_textures = [p for p in texture_paths if not os.path.exists(p)]
                    if missing_textures and self.cfg.mdl.auto_download:
                        logger.info(f"Downloading {len(missing_textures)} missing texture files...")
                        for texture_path in missing_textures:
                            try:
                                check_and_download_single(texture_path)
                            except Exception as e:
                                logger.warning(f"Failed to download texture {texture_path}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to process textures for {mdl_path}: {e}")

            # Apply the material
            try:
                self._apply_mdl_to_prim(mdl_path, env_prim_path)
            except Exception as e:
                logger.warning(f"Failed to apply MDL {mdl_path} to {env_prim_path}: {e}")

    def _select_mdl_path(self, available_paths: list[str]) -> str:
        """Select MDL path based on selection strategy using reproducible RNG."""
        if self.cfg.mdl.selection_strategy == "random":
            return self._rng.choice(available_paths)
        elif self.cfg.mdl.selection_strategy == "sequential":
            idx = self._mdl_selection_state["sequential_index"] % len(available_paths)
            self._mdl_selection_state["sequential_index"] += 1
            return available_paths[idx]
        elif self.cfg.mdl.selection_strategy == "weighted":
            # Get weights for available paths
            available_indices = [i for i, path in enumerate(self.cfg.mdl.mdl_paths) if path in available_paths]
            available_weights = [self.cfg.mdl.weights[i] for i in available_indices]
            return self._rng.choices(available_paths, weights=available_weights, k=1)[0]
        else:
            raise ValueError(f"Unknown selection strategy: {self.cfg.mdl.selection_strategy}")

    def _apply_mdl_to_prim(self, mdl_path: str, prim_path: str) -> None:
        """Apply MDL material to a specific prim.

        This is the internal implementation that was originally in material_util.py
        """
        # Convert to absolute path for IsaacSim
        mdl_path = os.path.abspath(mdl_path)

        if not os.path.exists(mdl_path):
            raise FileNotFoundError(f"Material file {mdl_path} does not exist")
        if not mdl_path.endswith(".mdl"):
            raise ValueError(f"Material file {mdl_path} must have .mdl extension")

        import isaacsim.core.utils.prims as prim_utils

        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim:
            raise ValueError(f"Prim not found at path {prim_path}")

        # Ensure UV coordinates first
        self._ensure_uv_for_hierarchy(prim)

        # Material name should match the MDL file basename (without .mdl extension)
        # Don't add _mat or timestamp - let IsaacSim use the exported material name from MDL
        mtl_name = os.path.basename(mdl_path).removesuffix(".mdl")
        import omni.kit.commands
        from omni.kit.material.library import get_material_prim_path
        from pxr import UsdShade

        _, mtl_prim_path = get_material_prim_path(mtl_name)

        logger.debug(f"Creating MDL material: {mtl_name} from {mdl_path}")

        success, result = omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=mdl_path,
            mtl_name=mtl_name,
            mtl_path=mtl_prim_path,
            select_new_prim=False,
        )
        if not success:
            logger.error(f"Failed to create material {mtl_name} at {mtl_prim_path}")
            raise RuntimeError(f"Failed to create material {mtl_name} at {mtl_prim_path}")

        logger.debug(f"Binding material {mtl_prim_path} to {prim.GetPath()}")

        success, result = omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim.GetPath(),
            material_path=mtl_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        if not success:
            logger.error(f"Failed to bind material at {mtl_prim_path} to {prim.GetPath()}")
            raise RuntimeError(f"Failed to bind material at {mtl_prim_path} to {prim.GetPath()}")

        logger.debug(f"Successfully applied MDL material {mtl_name} to {prim_path}")

    def _ensure_uv_for_hierarchy(self, prim, tile_scale: float = 1.0) -> None:
        """Ensure UV coordinates for all meshes in the prim hierarchy.

        Args:
            prim: USD prim to process
            tile_scale: Scale factor for UV tiling (passed to UV generation functions)
        """
        from pxr import Usd, UsdGeom

        # Process all meshes in the hierarchy
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Mesh):
                try:
                    self._ensure_uv_coordinates_improved(
                        UsdGeom.Mesh(p), tile=1.0 / tile_scale if tile_scale > 0 else 0.2
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate UV for mesh {p.GetPath()}: {e}")
            elif p.IsA(UsdGeom.Gprim) and not p.IsA(UsdGeom.Mesh):
                try:
                    self._ensure_basic_uv_for_gprim(p, tile_scale=tile_scale)
                except Exception as e:
                    logger.warning(f"Failed to generate UV for gprim {p.GetPath()}: {e}")

    def _ensure_uv_coordinates_improved(self, mesh, tile: float = 0.2) -> None:
        """Improved UV coordinate generation based on bounding box projection."""
        from math import isfinite

        from pxr import Gf, UsdGeom, Vt

        pvapi = UsdGeom.PrimvarsAPI(mesh)
        pv = pvapi.GetPrimvar("st")
        vals = pv.Get() if pv else None
        idx = mesh.GetFaceVertexIndicesAttr().Get() or []

        if pv and vals and len(vals) > 0:
            return  # UV coordinates already exist

        # Get mesh points
        pts = mesh.GetPointsAttr().Get() or []
        if not pts:
            return

        # Calculate bounding box and project to the largest two axes
        xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        rx = (max(xs) - min(xs)) if xs else 0
        ry = (max(ys) - min(ys)) if ys else 0
        rz = (max(zs) - min(zs)) if zs else 0

        # Sort axes by size to pick the largest two
        axes = sorted([("x", rx), ("y", ry), ("z", rz)], key=lambda t: t[1], reverse=True)
        a, b = axes[0][0], axes[1][0]

        def comp(p, axis):
            return p[0] if axis == "x" else (p[1] if axis == "y" else p[2])

        # Generate UV coordinates
        st_list = []
        for vid in idx:
            p = pts[vid]
            u = comp(p, a) * tile
            v = comp(p, b) * tile
            u = 0.0 if not isfinite(u) else u
            v = 0.0 if not isfinite(v) else v
            st_list.append((u, v))

        # Create UV primvar
        try:
            st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in st_list])
            api = UsdGeom.PrimvarsAPI(mesh)
            pv = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(st)
            pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
        except Exception as e:
            logger.warning(f"Failed to create UV coordinates for mesh: {e}")

    def _ensure_basic_uv_for_gprim(self, gprim, tile_scale: float = 1.0) -> None:
        """Enhanced UV coordinate generation for geometric primitives (Cube, Sphere, etc.).

        Args:
            gprim: The geometric primitive to generate UVs for
            tile_scale: Scale factor for UV tiling (smaller = more repetitions)
        """
        from pxr import Gf, Sdf, UsdGeom, Vt

        try:
            pvapi = UsdGeom.PrimvarsAPI(gprim)
            if pvapi.HasPrimvar("st"):
                return  # UV coordinates already exist

            prim_type = gprim.GetTypeName()

            if prim_type == "Cube":
                # Proper UV mapping for cube: 6 faces, 4 vertices per face = 24 faceVarying UVs
                # Each face gets proper 0-1 UV coordinates for correct texture mapping
                # Face order in USD Cube: -X, +X, -Y, +Y, -Z, +Z
                cube_uvs = []
                for face_idx in range(6):
                    # Each face gets a 0-1 UV square
                    cube_uvs.extend([
                        (0.0, 0.0),  # bottom-left
                        (tile_scale, 0.0),  # bottom-right
                        (tile_scale, tile_scale),  # top-right
                        (0.0, tile_scale),  # top-left
                    ])

                st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in cube_uvs])
                pv = pvapi.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                pv.Set(st)
                pv.SetInterpolation(UsdGeom.Tokens.faceVarying)
                logger.debug(f"Generated {len(cube_uvs)} faceVarying UVs for Cube with tile_scale={tile_scale}")
            else:
                # For other primitives, use simpler vertex-based UVs
                basic_uvs = [(0.0, 0.0), (tile_scale, 0.0), (tile_scale, tile_scale), (0.0, tile_scale)]

                st = Vt.Vec2fArray([Gf.Vec2f(u, v) for (u, v) in basic_uvs])
                pv = pvapi.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                pv.Set(st)
                pv.SetInterpolation(UsdGeom.Tokens.vertex)
                logger.debug(f"Generated {len(basic_uvs)} vertex UVs for {prim_type}")

        except Exception as e:
            logger.warning(f"Failed to create UV coordinates for {gprim.GetPath()}: {e}")

    def get_physical_properties(self) -> dict:
        """Get current physical properties for logging."""
        if not self.cfg.physical:
            return {}

        try:
            obj_inst = self._get_object_instance(self.cfg.obj_name)
            env_ids = self._get_env_ids()
            materials = obj_inst.root_physx_view.get_material_properties()

            return {
                "friction": materials[env_ids, :, 0].cpu().numpy(),
                "restitution": materials[env_ids, :, 2].cpu().numpy(),
            }
        except Exception:
            return {}

    def __call__(self) -> None:
        """Execute material randomization based on configuration.

        Randomization behavior depends on randomization_mode:
        - combined: Apply physics + best available visual (MDL > PBR)
        - physics_only: Apply only physical properties
        - visual_only: Apply only visual properties (MDL > PBR)
        - all_separate: Apply all enabled types independently
        """
        try:
            enabled_types = self._get_enabled_material_types()
            if not enabled_types:
                return

            if self.cfg.randomization_mode == "combined":
                self._apply_combined_materials(enabled_types)
            elif self.cfg.randomization_mode == "physics_only":
                self._apply_physics_only(enabled_types)
            elif self.cfg.randomization_mode == "visual_only":
                self._apply_visual_only(enabled_types)
            elif self.cfg.randomization_mode == "all_separate":
                self._apply_all_separate(enabled_types)
            else:
                raise ValueError(f"Unknown randomization mode: {self.cfg.randomization_mode}")

        except Exception as e:
            logger.error(f"Material randomization failed for {self.cfg.obj_name}: {e}")
            raise

    def _get_enabled_material_types(self) -> list[str]:
        """Get list of enabled material types."""
        enabled = []
        if self.cfg.physical and self.cfg.physical.enabled:
            enabled.append("physical")
        if self.cfg.pbr and self.cfg.pbr.enabled:
            enabled.append("pbr")
        if self.cfg.mdl and self.cfg.mdl.enabled:
            enabled.append("mdl")
        return enabled

    def _apply_combined_materials(self, enabled_types: list[str]) -> None:
        """Apply physics + best available visual material."""
        # Apply physical properties if enabled
        if "physical" in enabled_types:
            self.randomize_physical_properties()

        # Apply best available visual material (MDL > PBR)
        if "mdl" in enabled_types:
            self.randomize_mdl_material()
        elif "pbr" in enabled_types:
            self.randomize_pbr_properties()

    def _apply_physics_only(self, enabled_types: list[str]) -> None:
        """Apply only physical properties."""
        if "physical" in enabled_types:
            self.randomize_physical_properties()

    def _apply_visual_only(self, enabled_types: list[str]) -> None:
        """Apply only visual properties (MDL > PBR)."""
        if "mdl" in enabled_types:
            self.randomize_mdl_material()
        elif "pbr" in enabled_types:
            self.randomize_pbr_properties()

    def _apply_all_separate(self, enabled_types: list[str]) -> None:
        """Apply all enabled material types independently."""
        if "physical" in enabled_types:
            self.randomize_physical_properties()
        if "pbr" in enabled_types:
            self.randomize_pbr_properties()
        if "mdl" in enabled_types:
            self.randomize_mdl_material()
