"""Preset configurations for scene randomization.

This module provides curated material collections and preset scene configurations
for domain randomization, following the paper's methodology.
"""

from __future__ import annotations

from metasim.randomization.scene_randomizer import (
    SceneGeometryCfg,
    SceneMaterialPoolCfg,
    SceneRandomCfg,
)

# =============================================================================
# Scene Material Collections (from ARNOLD and vMaterials)
# =============================================================================


class SceneMaterialCollections:
    """Curated material collections for scene elements.

    Following the paper's methodology:
    - Table/Desktop: ~300 materials
    - Walls: ~150 materials
    - Floors: ~150 materials
    - Ceiling: subset of wall materials
    """

    @staticmethod
    def table_materials(
        base_path_arnold: str = "roboverse_data/materials/arnold",
        base_path_vmaterials: str = "roboverse_data/materials/vMaterials_2",
    ) -> list[str]:
        """Get curated table/desktop material paths (~300 materials).

        Combines wood, stone, and other suitable materials for tabletops.

        Args:
            base_path_arnold: Base path to ARNOLD materials
            base_path_vmaterials: Base path to vMaterials

        Returns:
            List of material file paths
        """
        materials = []

        # ARNOLD Wood materials (suitable for tables)
        arnold_wood = [
            "Wood/Ash.mdl",
            "Wood/Ash_Planks.mdl",
            "Wood/Bamboo.mdl",
            "Wood/Bamboo_Planks.mdl",
            "Wood/Beadboard.mdl",
            "Wood/Birch.mdl",
            "Wood/Birch_Planks.mdl",
            "Wood/Cherry.mdl",
            "Wood/Cherry_Planks.mdl",
            "Wood/Cork.mdl",
            "Wood/Mahogany.mdl",
            "Wood/Mahogany_Planks.mdl",
            "Wood/Oak.mdl",
            "Wood/Oak_Planks.mdl",
            "Wood/Parquet_Floor.mdl",
            "Wood/Plywood.mdl",
            "Wood/Timber.mdl",
            "Wood/Timber_Cladding.mdl",
            "Wood/Walnut.mdl",
            "Wood/Walnut_Planks.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_wood])

        # vMaterials Wood
        vmaterials_wood = [
            "Wood/Laminate_Oak.mdl",
            "Wood/OSB_Wood.mdl",
            "Wood/OSB_Wood_Splattered.mdl",
            "Wood/Wood_Bark.mdl",
            "Wood/Wood_Cork.mdl",
            "Wood/Wood_Tiles_Ash.mdl",
            "Wood/Wood_Tiles_Ash_Multicolor.mdl",
            "Wood/Wood_Tiles_Beech.mdl",
            "Wood/Wood_Tiles_Beech_Multicolor.mdl",
            "Wood/Wood_Tiles_Fineline.mdl",
            "Wood/Wood_Tiles_Fineline_Multicolor.mdl",
            "Wood/Wood_Tiles_Oak_Mountain.mdl",
            "Wood/Wood_Tiles_Oak_Mountain_Multicolor.mdl",
            "Wood/Wood_Tiles_Pine.mdl",
            "Wood/Wood_Tiles_Pine_Multicolor.mdl",
            "Wood/Wood_Tiles_Poplar.mdl",
            "Wood/Wood_Tiles_Poplar_Multicolor.mdl",
            "Wood/Wood_Tiles_Walnut.mdl",
            "Wood/Wood_Tiles_Walnut_Multicolor.mdl",
        ]
        materials.extend([f"{base_path_vmaterials}/{m}" for m in vmaterials_wood])

        # ARNOLD Stone/Masonry (suitable for tables)
        arnold_stone = [
            "Masonry/Adobe_Brick.mdl",
            "Masonry/Brick_Pavers.mdl",
            "Masonry/Concrete_Block.mdl",
            "Masonry/Brick_Wall_Brown.mdl",
            "Masonry/Brick_Wall_Red.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_stone])

        # vMaterials Stone
        vmaterials_stone = [
            "Stone/Granite_Polished.mdl",
            # "Stone/Marble_Polished.mdl",
            "Stone/Rosa_Beta.mdl",
            "Stone/Stone_Natural_Black.mdl",
            "Stone/Terrazzo.mdl",
        ]
        materials.extend([f"{base_path_vmaterials}/{m}" for m in vmaterials_stone])

        return materials

    @staticmethod
    def floor_materials(
        base_path_arnold: str = "roboverse_data/materials/arnold",
        base_path_vmaterials: str = "roboverse_data/materials/vMaterials_2",
    ) -> list[str]:
        """Get curated floor material paths (~150 materials).

        Combines carpet, wood, stone, and tile materials suitable for floors.

        Args:
            base_path_arnold: Base path to ARNOLD materials
            base_path_vmaterials: Base path to vMaterials

        Returns:
            List of material file paths
        """
        materials = []

        # ARNOLD Carpet materials
        arnold_carpet = [
            "Carpet/Carpet_Beige.mdl",
            "Carpet/Carpet_Berber_Gray.mdl",
            "Carpet/Carpet_Berber_Multi.mdl",
            "Carpet/Carpet_Charcoal.mdl",
            "Carpet/Carpet_Cream.mdl",
            "Carpet/Carpet_Diamond_Yellow.mdl",
            "Carpet/Carpet_Forest.mdl",
            "Carpet/Carpet_Gray.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_carpet])

        # ARNOLD Wood flooring
        arnold_wood_floor = [
            "Wood/Parquet_Floor.mdl",
            "Wood/Ash_Planks.mdl",
            "Wood/Bamboo_Planks.mdl",
            "Wood/Birch_Planks.mdl",
            "Wood/Cherry_Planks.mdl",
            "Wood/Oak_Planks.mdl",
            "Wood/Mahogany_Planks.mdl",
            "Wood/Walnut_Planks.mdl",
            "Wood/Beadboard.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_wood_floor])

        # vMaterials Wood flooring
        vmaterials_wood_floor = [
            "Wood/Laminate_Oak.mdl",
            "Wood/Wood_Tiles_Ash.mdl",
            "Wood/Wood_Tiles_Beech.mdl",
            "Wood/Wood_Tiles_Fineline.mdl",
            "Wood/Wood_Tiles_Oak_Mountain.mdl",
            "Wood/Wood_Tiles_Pine.mdl",
            "Wood/Wood_Tiles_Poplar.mdl",
            "Wood/Wood_Tiles_Walnut.mdl",
        ]
        materials.extend([f"{base_path_vmaterials}/{m}" for m in vmaterials_wood_floor])

        # ARNOLD Masonry for floors
        arnold_masonry = [
            "Masonry/Brick_Pavers.mdl",
            "Masonry/Concrete_Block.mdl",
            "Masonry/Brick_Wall_Brown.mdl",
            "Masonry/Brick_Wall_Red.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_masonry])

        # vMaterials Stone for floors
        vmaterials_stone = [
            "Stone/Terrazzo.mdl",
        ]
        materials.extend([f"{base_path_vmaterials}/{m}" for m in vmaterials_stone])

        return materials

    @staticmethod
    def wall_materials(
        base_path_arnold: str = "roboverse_data/materials/arnold",
        base_path_vmaterials: str = "roboverse_data/materials/vMaterials_2",
    ) -> list[str]:
        """Get curated wall material paths (~150 materials).

        Combines architecture materials, painted surfaces, and suitable textures.

        Args:
            base_path_arnold: Base path to ARNOLD materials
            base_path_vmaterials: Base path to vMaterials

        Returns:
            List of material file paths
        """
        materials = []

        # ARNOLD Architecture materials
        arnold_architecture = [
            "Architecture/Ceiling_Tiles.mdl",
            "Architecture/Roof_Tiles.mdl",
            "Architecture/Shingles_01.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_architecture])

        # ARNOLD Masonry for walls
        arnold_masonry = [
            "Masonry/Adobe_Brick.mdl",
            "Masonry/Brick_Wall_Brown.mdl",
            "Masonry/Brick_Wall_Red.mdl",
            "Masonry/Concrete_Block.mdl",
            "Masonry/Brick_Pavers.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_masonry])

        # ARNOLD Wood paneling for walls
        arnold_wood_wall = [
            "Wood/Beadboard.mdl",
            "Wood/Timber_Cladding.mdl",
            "Wood/Plywood.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_wood_wall])

        return materials

    @staticmethod
    def ceiling_materials(base_path_arnold: str = "roboverse_data/materials/arnold") -> list[str]:
        """Get curated ceiling material paths (subset of wall materials).

        Args:
            base_path_arnold: Base path to ARNOLD materials

        Returns:
            List of material file paths
        """
        materials = []

        # ARNOLD Architecture materials for ceilings
        arnold_ceiling = [
            "Architecture/Ceiling_Tiles.mdl",
            "Architecture/Shingles_01.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_ceiling])

        # Some wood paneling
        arnold_wood = [
            "Wood/Beadboard.mdl",
            "Wood/Plywood.mdl",
        ]
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_wood])

        # Simple masonry
        arnold_masonry = []
        materials.extend([f"{base_path_arnold}/{m}" for m in arnold_masonry])

        return materials


# =============================================================================
# Preset Scene Configurations
# =============================================================================


class ScenePresets:
    """Pre-configured scene setups for common scenarios."""

    @staticmethod
    def empty_room(room_size: float = 5.0, wall_height: float = 3.0, wall_thickness: float = 0.1) -> SceneRandomCfg:
        """Create an empty room with floor, walls, and ceiling.

        Args:
            room_size: Size of the room (square)
            wall_height: Height of walls
            wall_thickness: Thickness of walls

        Returns:
            Scene randomization configuration
        """
        return SceneRandomCfg(
            floor=SceneGeometryCfg(
                enabled=True,
                size=(room_size, room_size, wall_thickness),
                position=(0.0, 0.0, 0.005),  # Slightly above z=0 to avoid z-fighting with IsaacSim default ground
                material_randomization=True,
            ),
            walls=SceneGeometryCfg(
                enabled=True,
                size=(room_size, wall_thickness, wall_height),
                position=(0.0, 0.0, wall_height / 2),
                material_randomization=True,
            ),
            ceiling=SceneGeometryCfg(
                enabled=True,
                size=(room_size, room_size, wall_thickness),
                position=(0.0, 0.0, wall_height + wall_thickness / 2),
                material_randomization=True,
            ),
            floor_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.floor_materials(),
                selection_strategy="random",
            ),
            wall_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.wall_materials(),
                selection_strategy="random",
            ),
            ceiling_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.ceiling_materials(),
                selection_strategy="random",
            ),
            only_if_no_scene=True,
        )

    @staticmethod
    def tabletop_workspace(
        room_size: float = 5.0,
        wall_height: float = 3.0,
        table_size: tuple[float, float, float] = (1.5, 1.0, 0.05),
        table_height: float = 0.75,
    ) -> SceneRandomCfg:
        """Create a tabletop manipulation workspace.

        Args:
            room_size: Size of the room (square)
            wall_height: Height of walls
            table_size: Size of the table (x, y, z)
            table_height: Height of table surface from ground

        Returns:
            Scene randomization configuration
        """
        wall_thickness = 0.1

        return SceneRandomCfg(
            floor=SceneGeometryCfg(
                enabled=True,
                size=(room_size, room_size, wall_thickness),
                position=(0.0, 0.0, 0.005),  # Slightly above z=0 to avoid z-fighting
                material_randomization=True,
            ),
            walls=SceneGeometryCfg(
                enabled=True,
                size=(room_size, wall_thickness, wall_height),
                position=(0.0, 0.0, wall_height / 2),
                material_randomization=True,
            ),
            ceiling=SceneGeometryCfg(
                enabled=True,
                size=(room_size, room_size, wall_thickness),
                position=(0.0, 0.0, wall_height + wall_thickness / 2),
                material_randomization=True,
            ),
            table=SceneGeometryCfg(
                enabled=True,
                size=table_size,
                position=(0.0, 0.0, table_height - table_size[2] / 2),
                material_randomization=True,
            ),
            floor_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.floor_materials(),
                selection_strategy="random",
            ),
            wall_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.wall_materials(),
                selection_strategy="random",
            ),
            ceiling_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.ceiling_materials(),
                selection_strategy="random",
            ),
            table_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.table_materials(),
                selection_strategy="random",
            ),
            only_if_no_scene=True,
        )

    @staticmethod
    def floor_only(floor_size: float = 10.0, floor_thickness: float = 0.1) -> SceneRandomCfg:
        """Create only a floor (minimal scene).

        Args:
            floor_size: Size of the floor (square)
            floor_thickness: Thickness of floor

        Returns:
            Scene randomization configuration
        """
        return SceneRandomCfg(
            floor=SceneGeometryCfg(
                enabled=True,
                size=(floor_size, floor_size, floor_thickness),
                position=(0.0, 0.0, 0.005),  # Slightly above z=0 to avoid z-fighting
                material_randomization=True,
            ),
            floor_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.floor_materials(),
                selection_strategy="random",
            ),
            only_if_no_scene=True,
        )

    @staticmethod
    def custom_scene(
        floor_cfg: SceneGeometryCfg | None = None,
        walls_cfg: SceneGeometryCfg | None = None,
        ceiling_cfg: SceneGeometryCfg | None = None,
        table_cfg: SceneGeometryCfg | None = None,
        floor_materials: list[str] | None = None,
        wall_materials: list[str] | None = None,
        ceiling_materials: list[str] | None = None,
        table_materials: list[str] | None = None,
        only_if_no_scene: bool = True,
    ) -> SceneRandomCfg:
        """Create a fully customizable scene configuration.

        Args:
            floor_cfg: Floor geometry configuration
            walls_cfg: Walls geometry configuration
            ceiling_cfg: Ceiling geometry configuration
            table_cfg: Table geometry configuration
            floor_materials: Custom floor materials (if None, uses defaults)
            wall_materials: Custom wall materials (if None, uses defaults)
            ceiling_materials: Custom ceiling materials (if None, uses defaults)
            table_materials: Custom table materials (if None, uses defaults)
            only_if_no_scene: Only create if no predefined scene exists

        Returns:
            Scene randomization configuration
        """
        # Use default materials if not provided
        if floor_materials is None:
            floor_materials = SceneMaterialCollections.floor_materials()
        if wall_materials is None:
            wall_materials = SceneMaterialCollections.wall_materials()
        if ceiling_materials is None:
            ceiling_materials = SceneMaterialCollections.ceiling_materials()
        if table_materials is None:
            table_materials = SceneMaterialCollections.table_materials()

        return SceneRandomCfg(
            floor=floor_cfg,
            walls=walls_cfg,
            ceiling=ceiling_cfg,
            table=table_cfg,
            floor_materials=SceneMaterialPoolCfg(
                material_paths=floor_materials,
                selection_strategy="random",
            )
            if floor_cfg is not None
            else None,
            wall_materials=SceneMaterialPoolCfg(
                material_paths=wall_materials,
                selection_strategy="random",
            )
            if walls_cfg is not None
            else None,
            ceiling_materials=SceneMaterialPoolCfg(
                material_paths=ceiling_materials,
                selection_strategy="random",
            )
            if ceiling_cfg is not None
            else None,
            table_materials=SceneMaterialPoolCfg(
                material_paths=table_materials,
                selection_strategy="random",
            )
            if table_cfg is not None
            else None,
            only_if_no_scene=only_if_no_scene,
        )
