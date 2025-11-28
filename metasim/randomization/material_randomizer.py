"""Material Randomizer - Property editor for object materials.

The MaterialRandomizer modifies material properties of existing objects.
It supports both Static Objects (Handler-managed) and Dynamic Objects (SceneRandomizer-managed)
through unified access via ObjectRegistry.

Supported material types:
- MDL: Material Definition Language files (Arnold/OmniPBR)
- PBR: Physically Based Rendering properties (roughness, metallic, etc.)
- Physical: Physics material properties (friction, restitution)

Key features:
- Unified access to all objects via ObjectRegistry
- Automatic material variant randomization for diversity
- Auto-download support for remote materials
- Supports Hybrid simulation (uses render_handler)
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

import torch
from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.randomization.core.isaacsim_adapter import IsaacSimAdapter
from metasim.randomization.core.object_registry import ObjectRegistry
from metasim.utils.configclass import configclass

# =============================================================================
# Utility Functions
# =============================================================================


def list_materials_in_mdl(mdl_file_path: str) -> list[str]:
    """List all material names in an MDL file.

    Args:
        mdl_file_path: Path to MDL file

    Returns:
        List of material names (export materials first)
    """
    if not os.path.exists(mdl_file_path):
        return []

    try:
        with open(mdl_file_path, encoding="utf-8") as f:
            content = f.read()

        import re

        # Find export materials (preferred)
        export_pattern = r"export\s+material\s+(\w+)\s*\("
        export_matches = re.findall(export_pattern, content)

        # Find all materials
        all_pattern = r"(?:export\s+)?material\s+(\w+)\s*\("
        all_matches = re.findall(all_pattern, content)

        # Return unique, with export first
        seen = set()
        result = []
        for mat in export_matches + all_matches:
            if mat not in seen:
                seen.add(mat)
                result.append(mat)

        return result

    except Exception as e:
        logger.warning(f"Failed to list materials from {mdl_file_path}: {e}")
        return []


# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class PhysicalMaterialCfg:
    """Physical material properties configuration.

    Attributes:
        friction_range: Friction coefficient range (min, max)
        restitution_range: Restitution (bounciness) range (min, max)
        distribution: Random sampling distribution
        enabled: Whether to apply physical material randomization
    """

    friction_range: tuple[float, float] | None = None
    restitution_range: tuple[float, float] | None = None
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
    enabled: bool = True


@configclass
class PBRMaterialCfg:
    """PBR material properties configuration.

    Attributes:
        roughness_range: Surface roughness range (0=smooth, 1=rough)
        metallic_range: Metallic property range (0=dielectric, 1=metallic)
        specular_range: Specular reflection intensity range
        diffuse_color_range: RGB color ranges
        distribution: Random sampling distribution
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
    """MDL material configuration.

    Attributes:
        mdl_paths: List of MDL file paths (supports "path.mdl::MaterialName" syntax)
        selection_strategy: Material selection strategy
        weights: Weights for weighted selection
        randomize_material_variant: Randomly select material variants within MDL files
        enabled: Whether to apply MDL randomization
        auto_download: Auto-download missing MDL files
        validate_paths: Validate file existence at init
    """

    mdl_paths: list[str] = dataclasses.field(default_factory=list)
    selection_strategy: Literal["random", "sequential", "weighted"] = "random"
    weights: list[float] | None = None
    randomize_material_variant: bool = True
    enabled: bool = True
    auto_download: bool = True
    validate_paths: bool = True

    def __post_init__(self):
        if self.enabled and not self.mdl_paths:
            logger.warning("MDL randomization enabled but no paths provided")

        if self.mdl_paths and self.selection_strategy == "weighted":
            if self.weights is None or len(self.weights) != len(self.mdl_paths):
                raise ValueError("Weights must match mdl_paths length for weighted selection")


@configclass
class MaterialRandomCfg:
    """Unified material randomization configuration.

    Attributes:
        obj_name: Name of object to randomize (must exist in ObjectRegistry)
        physical: Physical material properties configuration
        pbr: PBR material properties configuration
        mdl: MDL material configuration
        env_ids: Environment IDs to apply randomization (None = all)
    """

    obj_name: str = dataclasses.MISSING
    physical: PhysicalMaterialCfg | None = None
    pbr: PBRMaterialCfg | None = None
    mdl: MDLMaterialCfg | None = None
    env_ids: list[int] | None = None

    def __post_init__(self):
        # Ensure at least one material type is enabled
        has_enabled = False
        if self.physical and self.physical.enabled:
            has_enabled = True
        if self.pbr and self.pbr.enabled:
            has_enabled = True
        if self.mdl and self.mdl.enabled:
            has_enabled = True

        if not has_enabled:
            raise ValueError("At least one material randomization type must be enabled")


# =============================================================================
# Material Randomizer Implementation
# =============================================================================


class MaterialRandomizer(BaseRandomizerType):
    """Material randomizer for all objects.

    Responsibilities:
    - Modify material properties of existing objects
    - Support Static Objects (Handler-managed) and Dynamic Objects (Scene-managed)
    - NOT responsible for: Creating/deleting objects

    Characteristics:
    - Uses ObjectRegistry to find objects (supports all object types)
    - Uses IsaacSimAdapter for material application
    - Supports MDL, PBR, and Physical materials
    - Hybrid support: uses render_handler

    Usage:
        randomizer = MaterialRandomizer(
            MaterialRandomCfg(
                obj_name="table",  # Can be Static or Dynamic object
                mdl=MDLMaterialCfg(mdl_paths=["materials/wood/*.mdl"])
            ),
            seed=42
        )
        randomizer.bind_handler(handler)
        randomizer()  # Apply material randomization
    """

    REQUIRES_HANDLER = "render"  # Use render_handler for Hybrid

    def __init__(self, cfg: MaterialRandomCfg, seed: int | None = None):
        """Initialize material randomizer.

        Args:
            cfg: Material randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.cfg = cfg
        self.registry: ObjectRegistry | None = None
        self.adapter: IsaacSimAdapter | None = None

        # Sequential selection state
        self._sequential_index: int = 0

        # Torch generator for tensor operations
        self._torch_generator: torch.Generator | None = None

    def __call__(self):
        """Execute material randomization.

        Note: MDL and PBR are mutually exclusive. If both are enabled,
        MDL takes priority (as it's more realistic).
        """
        # Visual material: MDL takes priority over PBR
        if self.cfg.mdl and self.cfg.mdl.enabled:
            self.randomize_mdl_material()
        elif self.cfg.pbr and self.cfg.pbr.enabled:
            self.randomize_pbr_material()

        # Physical material (can coexist with visual materials)
        if self.cfg.physical and self.cfg.physical.enabled:
            self.randomize_physical_material()

        # Flush visual updates for instant switching
        self._flush_visual_updates()

    # -------------------------------------------------------------------------
    # MDL Material Randomization
    # -------------------------------------------------------------------------

    def randomize_mdl_material(self):
        """Apply MDL material randomization."""
        if not self.cfg.mdl or not self.cfg.mdl.mdl_paths:
            return

        env_ids = self.cfg.env_ids or list(range(self._actual_handler.num_envs))

        # Get prim paths from Registry (supports both Static and Dynamic objects)
        try:
            prim_paths = self.registry.get_prim_paths(self.cfg.obj_name, env_ids)
        except ValueError as e:
            logger.error(f"MaterialRandomizer: {e}")
            return

        # Apply material to each prim
        applied_prims = []
        for prim_path in prim_paths:
            # Select MDL
            mdl_path = self._select_mdl_path()
            mdl_path = os.path.abspath(mdl_path)

            # Select material variant
            material_name = None
            if self.cfg.mdl.randomize_material_variant and "::" not in mdl_path:
                materials = list_materials_in_mdl(mdl_path)
                if len(materials) > 1:
                    material_name = self.rng.choice(materials)
            elif "::" in mdl_path:
                mdl_path, material_name = mdl_path.split("::", 1)

            # Apply MDL (Adapter handles download automatically)
            try:
                self.adapter.apply_mdl_material(
                    prim_path, mdl_path, material_name, auto_download=self.cfg.mdl.auto_download
                )
                applied_prims.append(prim_path)  # Track successful applications
            except Exception as e:
                logger.warning(f"Failed to apply MDL to {prim_path}: {e}")

            self._mark_visual_dirty()

        # Force pose nudge on successfully applied prims (IsaacSim 4.5+ requirement)
        if applied_prims and hasattr(self.adapter, "force_pose_nudge"):
            try:
                self.adapter.force_pose_nudge(applied_prims)
            except Exception as e:
                logger.debug(f"Pose nudge failed (non-critical): {e}")

    def _select_mdl_path(self) -> str:
        """Select MDL path based on selection strategy."""
        if self.cfg.mdl.selection_strategy == "random":
            return self.rng.choice(self.cfg.mdl.mdl_paths)
        elif self.cfg.mdl.selection_strategy == "sequential":
            idx = self._sequential_index % len(self.cfg.mdl.mdl_paths)
            self._sequential_index += 1
            return self.cfg.mdl.mdl_paths[idx]
        elif self.cfg.mdl.selection_strategy == "weighted":
            return self.rng.choices(self.cfg.mdl.mdl_paths, weights=self.cfg.mdl.weights, k=1)[0]
        else:
            return self.cfg.mdl.mdl_paths[0]

    # -------------------------------------------------------------------------
    # PBR Material Randomization
    # -------------------------------------------------------------------------

    def randomize_pbr_material(self):
        """Apply PBR material randomization."""
        if not self.cfg.pbr:
            return

        env_ids = self.cfg.env_ids or list(range(self._actual_handler.num_envs))

        # Get prim paths from Registry
        try:
            prim_paths = self.registry.get_prim_paths(self.cfg.obj_name, env_ids)
        except ValueError as e:
            logger.error(f"MaterialRandomizer: {e}")
            return

        # Generate PBR config
        pbr_config = {}

        if self.cfg.pbr.roughness_range:
            pbr_config["roughness"] = self._generate_random_value(
                self.cfg.pbr.roughness_range, self.cfg.pbr.distribution
            )

        if self.cfg.pbr.metallic_range:
            pbr_config["metallic"] = self._generate_random_value(self.cfg.pbr.metallic_range, self.cfg.pbr.distribution)

        if self.cfg.pbr.specular_range:
            pbr_config["specular"] = self._generate_random_value(self.cfg.pbr.specular_range, self.cfg.pbr.distribution)

        if self.cfg.pbr.diffuse_color_range:
            pbr_config["diffuse_color"] = tuple(
                self._generate_random_value(r, self.cfg.pbr.distribution) for r in self.cfg.pbr.diffuse_color_range
            )

        # Apply PBR to each prim
        for prim_path in prim_paths:
            try:
                self.adapter.apply_pbr_material(prim_path, pbr_config)
            except Exception as e:
                logger.warning(f"Failed to apply PBR to {prim_path}: {e}")

        self._mark_visual_dirty()

    # -------------------------------------------------------------------------
    # Physical Material Randomization
    # -------------------------------------------------------------------------

    def randomize_physical_material(self):
        """Apply physical material randomization.

        Note: This only works for Static Objects with physics.
        Dynamic Objects (pure visual) will skip physics randomization.
        """
        if not self.cfg.physical:
            return

        # Check if object has physics
        if not self.registry.has_physics(self.cfg.obj_name):
            logger.debug(
                f"MaterialRandomizer: Object '{self.cfg.obj_name}' has no physics. "
                f"Physical material randomization will be skipped."
            )
            return

        # Get object from Handler (must be Static object)
        try:
            if self.cfg.obj_name in self._actual_handler.scene.articulations:
                obj_inst = self._actual_handler.scene.articulations[self.cfg.obj_name]
            elif self.cfg.obj_name in self._actual_handler.scene.rigid_objects:
                obj_inst = self._actual_handler.scene.rigid_objects[self.cfg.obj_name]
            else:
                logger.error(
                    f"Static object '{self.cfg.obj_name}' not found in Handler.scene. "
                    f"Physical material randomization only works for Handler-managed objects."
                )
                return
        except AttributeError:
            logger.error("Handler does not have scene attribute (not IsaacSim?)")
            return

        env_ids = self.cfg.env_ids or list(range(self._actual_handler.num_envs))
        # Convert env_ids to tensor for IsaacLab API (device will be matched per-operation)
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.int32)
        num_envs = env_ids_tensor.shape[0]

        # Randomize friction
        if self.cfg.physical.friction_range:
            new_friction = self._generate_random_tensor(
                num_envs, self.cfg.physical.friction_range, self.cfg.physical.distribution
            )

            try:
                # Get current material properties
                materials = obj_inst.root_physx_view.get_material_properties()

                # Update friction (index 0=static, 1=dynamic, 2=restitution)
                if len(materials.shape) == 3:
                    # [num_envs, num_bodies, 3]
                    materials[env_ids_tensor, :, 0] = new_friction.unsqueeze(1).to(materials.device)
                    materials[env_ids_tensor, :, 1] = new_friction.unsqueeze(1).to(materials.device)
                else:
                    # [num_envs, 3]
                    materials[env_ids_tensor, 0] = new_friction.to(materials.device)
                    materials[env_ids_tensor, 1] = new_friction.to(materials.device)

                # Set back
                obj_inst.root_physx_view.set_material_properties(materials, env_ids_tensor)
            except Exception as e:
                logger.error(f"Failed to set friction: {e}")

        # Randomize restitution
        if self.cfg.physical.restitution_range:
            new_restitution = self._generate_random_tensor(
                num_envs, self.cfg.physical.restitution_range, self.cfg.physical.distribution
            )

            try:
                # Get current material properties
                materials = obj_inst.root_physx_view.get_material_properties()

                # Update restitution (index 2)
                if len(materials.shape) == 3:
                    # [num_envs, num_bodies, 3]
                    materials[env_ids_tensor, :, 2] = new_restitution.unsqueeze(1).to(materials.device)
                else:
                    # [num_envs, 3]
                    materials[env_ids_tensor, 2] = new_restitution.to(materials.device)

                # Set back
                obj_inst.root_physx_view.set_material_properties(materials, env_ids_tensor)
            except Exception as e:
                logger.error(f"Failed to set restitution: {e}")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _generate_random_value(self, value_range: tuple[float, float], distribution: str) -> float:
        """Generate a single random value.

        Args:
            value_range: Value range (min, max)
            distribution: Distribution type

        Returns:
            Random value
        """
        if distribution == "uniform":
            return self.rng.uniform(value_range[0], value_range[1])
        elif distribution == "log_uniform":
            log_min = torch.log(torch.tensor(value_range[0])).item()
            log_max = torch.log(torch.tensor(value_range[1])).item()
            return torch.exp(torch.tensor(self.rng.uniform(log_min, log_max))).item()
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            val = self.rng.gauss(mean, std)
            return max(value_range[0], min(value_range[1], val))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _generate_random_tensor(self, size: int, value_range: tuple[float, float], distribution: str) -> torch.Tensor:
        """Generate random tensor.

        Args:
            size: Tensor size
            value_range: Value range (min, max)
            distribution: Distribution type

        Returns:
            Random tensor
        """
        generator = self._torch_generator or torch.Generator()

        if distribution == "uniform":
            rand_vals = torch.rand(size, generator=generator)
            return rand_vals * (value_range[1] - value_range[0]) + value_range[0]
        elif distribution == "log_uniform":
            log_min = torch.log(torch.tensor(value_range[0]))
            log_max = torch.log(torch.tensor(value_range[1]))
            rand_vals = torch.rand(size, generator=generator)
            return torch.exp(rand_vals * (log_max - log_min) + log_min)
        elif distribution == "gaussian":
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            return torch.clamp(torch.randn(size, generator=generator) * std + mean, value_range[0], value_range[1])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def _flush_visual_updates(self):
        """Flush visual updates to ensure materials are visible instantly.

        This is critical for real-time material switching to be visible.
        Respects both instance-level and global defer flags for atomic multi-randomizer operations.
        """
        # Check if flush is deferred (for atomic multi-randomizer updates)
        if hasattr(self, "_defer_visual_flush") and self._defer_visual_flush:
            return  # Skip flush, will be done by caller

        # Check global defer flag (set by apply_randomization for 22→1 flush optimization)
        if (
            hasattr(self._actual_handler, "_defer_all_visual_flushes")
            and self._actual_handler._defer_all_visual_flushes
        ):
            return  # Skip flush, will be done by apply_randomization

        if hasattr(self._actual_handler, "flush_visual_updates"):
            self._actual_handler.flush_visual_updates()
