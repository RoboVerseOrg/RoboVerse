"""Material randomization presets and utilities.

Provides common material configurations while allowing full customization.
"""

from __future__ import annotations

from ..material_randomizer import MaterialRandomCfg, MDLMaterialCfg, PBRMaterialCfg, PhysicalMaterialCfg

# =============================================================================
# Common Material Property Ranges
# =============================================================================


class MaterialProperties:
    """Common material property ranges for realistic randomization."""

    # Physical properties
    FRICTION_LOW = (0.1, 0.3)  # Ice, smooth plastic
    FRICTION_MEDIUM = (0.4, 0.8)  # Wood, concrete
    FRICTION_HIGH = (0.9, 1.5)  # Rubber, rough surfaces

    RESTITUTION_LOW = (0.0, 0.3)  # Clay, soft materials
    RESTITUTION_MEDIUM = (0.4, 0.7)  # Wood, plastic
    RESTITUTION_HIGH = (0.8, 0.95)  # Rubber balls, bouncy materials

    # PBR properties
    ROUGHNESS_SMOOTH = (0.0, 0.2)  # Polished metal, glass
    ROUGHNESS_MEDIUM = (0.3, 0.7)  # Painted surfaces, plastic
    ROUGHNESS_ROUGH = (0.8, 1.0)  # Concrete, unfinished wood

    METALLIC_NON = (0.0, 0.0)  # Dielectric materials
    METALLIC_PARTIAL = (0.0, 0.3)  # Mixed materials
    METALLIC_FULL = (0.8, 1.0)  # Pure metals

    # Color ranges
    COLOR_FULL = ((0.2, 1.0), (0.2, 1.0), (0.2, 1.0))  # Full spectrum (brighter)
    COLOR_BRIGHT = ((0.5, 1.0), (0.5, 1.0), (0.5, 1.0))  # Very bright colors
    COLOR_WARM = ((0.7, 1.0), (0.3, 0.8), (0.0, 0.4))  # Reds, oranges, yellows
    COLOR_COOL = ((0.0, 0.4), (0.3, 0.8), (0.7, 1.0))  # Blues, greens, purples
    COLOR_NEUTRAL = ((0.3, 0.8), (0.3, 0.8), (0.3, 0.8))  # Grays, browns


# =============================================================================
# MDL Material Collections
# =============================================================================


class MDLCollections:
    """Collections of MDL material paths organized by type."""

    @staticmethod
    def wood_materials(base_path: str = "roboverse_data/materials/arnold/Wood") -> list[str]:
        """Get wood material paths."""
        materials = [
            "Ash.mdl",
            "Bamboo.mdl",
            "Birch.mdl",
            "Cherry.mdl",
            "Oak.mdl",
            "Walnut.mdl",
            "Mahogany.mdl",
            "Cork.mdl",
            "Ash_Planks.mdl",
            "Bamboo_Planks.mdl",
            "Birch_Planks.mdl",
            "Cherry_Planks.mdl",
            "Mahogany_Planks.mdl",
            "Oak_Planks.mdl",
        ]
        return [f"{base_path}/{m}" for m in materials]

    @staticmethod
    def metal_materials(base_path: str = "roboverse_data/materials/vMaterials_2/Metal") -> list[str]:
        """Get metal material paths."""
        materials = [
            "Aluminum_Brushed.mdl",
            "Bronze_Polished.mdl",
            "Copper_Brushed.mdl",
            "Gold_Brushed.mdl",
            "Silver_Brushed.mdl",
            "Stainless_Steel_Brushed.mdl",
            "Brass_Brushed.mdl",
            "Iron_Brushed.mdl",
        ]
        return [f"{base_path}/{m}" for m in materials]

    @staticmethod
    def carpet_materials(base_path: str = "roboverse_data/materials/arnold/Carpet") -> list[str]:
        """Get carpet/fabric material paths."""
        materials = [
            "Carpet_Beige.mdl",
            "Carpet_Charcoal.mdl",
            "Carpet_Gray.mdl",
            "Carpet_Forest.mdl",
            "Carpet_Cream.mdl",
        ]
        return [f"{base_path}/{m}" for m in materials]

    @staticmethod
    def stone_materials(base_path: str = "roboverse_data/materials/vMaterials_2/Stone") -> list[str]:
        """Get stone/concrete material paths."""
        materials = ["Granite_Polished.mdl", "Stone_Natural_Black.mdl", "Terrazzo.mdl", "Rosa_Beta.mdl"]
        return [f"{base_path}/{m}" for m in materials]


# =============================================================================
# Preset Material Configurations
# =============================================================================


class MaterialPresets:
    """Pre-configured material setups for common scenarios."""

    @staticmethod
    def plastic_object(
        obj_name: str, color_range: tuple = MaterialProperties.COLOR_BRIGHT, randomization_mode: str = "combined"
    ) -> MaterialRandomCfg:
        """Create plastic material configuration."""
        return MaterialRandomCfg(
            obj_name=obj_name,
            pbr=PBRMaterialCfg(
                roughness_range=MaterialProperties.ROUGHNESS_SMOOTH,
                metallic_range=MaterialProperties.METALLIC_NON,
                diffuse_color_range=color_range,
                enabled=True,
            ),
            physical=PhysicalMaterialCfg(
                friction_range=MaterialProperties.FRICTION_LOW,
                restitution_range=MaterialProperties.RESTITUTION_MEDIUM,
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def rubber_object(
        obj_name: str, color_range: tuple = MaterialProperties.COLOR_NEUTRAL, randomization_mode: str = "combined"
    ) -> MaterialRandomCfg:
        """Create rubber material configuration."""
        return MaterialRandomCfg(
            obj_name=obj_name,
            pbr=PBRMaterialCfg(
                roughness_range=MaterialProperties.ROUGHNESS_ROUGH,
                metallic_range=MaterialProperties.METALLIC_NON,
                diffuse_color_range=color_range,
                enabled=True,
            ),
            physical=PhysicalMaterialCfg(
                friction_range=MaterialProperties.FRICTION_HIGH,
                restitution_range=MaterialProperties.RESTITUTION_HIGH,
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

    @staticmethod
    def metal_object(
        obj_name: str,
        use_mdl: bool = True,
        mdl_base_path: str = "roboverse_data/materials/vMaterials_2/Metal",
        randomization_mode: str = "combined",
    ) -> MaterialRandomCfg:
        """Create metal material configuration."""
        config = MaterialRandomCfg(
            obj_name=obj_name,
            physical=PhysicalMaterialCfg(
                friction_range=MaterialProperties.FRICTION_MEDIUM,
                restitution_range=MaterialProperties.RESTITUTION_LOW,
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

        if use_mdl:
            config.mdl = MDLMaterialCfg(mdl_paths=MDLCollections.metal_materials(mdl_base_path), enabled=True)
        else:
            config.pbr = PBRMaterialCfg(
                roughness_range=MaterialProperties.ROUGHNESS_SMOOTH,
                metallic_range=MaterialProperties.METALLIC_FULL,
                diffuse_color_range=((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
                enabled=True,
            )

        return config

    @staticmethod
    def wood_object(
        obj_name: str,
        use_mdl: bool = True,
        mdl_base_path: str = "roboverse_data/materials/arnold/Wood",
        randomization_mode: str = "combined",
    ) -> MaterialRandomCfg:
        """Create wood material configuration."""
        config = MaterialRandomCfg(
            obj_name=obj_name,
            physical=PhysicalMaterialCfg(
                friction_range=MaterialProperties.FRICTION_MEDIUM,
                restitution_range=MaterialProperties.RESTITUTION_MEDIUM,
                enabled=True,
            ),
            randomization_mode=randomization_mode,
        )

        if use_mdl:
            config.mdl = MDLMaterialCfg(mdl_paths=MDLCollections.wood_materials(mdl_base_path), enabled=True)
        else:
            config.pbr = PBRMaterialCfg(
                roughness_range=MaterialProperties.ROUGHNESS_MEDIUM,
                metallic_range=MaterialProperties.METALLIC_NON,
                diffuse_color_range=((0.3, 0.8), (0.2, 0.6), (0.1, 0.4)),  # Brown tones
                enabled=True,
            )

        return config

    @staticmethod
    def custom_object(
        obj_name: str,
        physical_config: PhysicalMaterialCfg | None = None,
        pbr_config: PBRMaterialCfg | None = None,
        mdl_config: MDLMaterialCfg | None = None,
        randomization_mode: str = "combined",
    ) -> MaterialRandomCfg:
        """Create fully customizable material configuration."""
        return MaterialRandomCfg(
            obj_name=obj_name,
            physical=physical_config,
            pbr=pbr_config,
            mdl=mdl_config,
            randomization_mode=randomization_mode,
        )
