"""Preset configurations for scene randomization.

This module provides curated material collections and preset scene configurations
for domain randomization, following the paper's methodology.
"""

from __future__ import annotations

from metasim.randomization.scene_randomizer import SceneGeometryCfg, SceneMaterialPoolCfg, SceneRandomCfg

from .material_presets import MDLCollections

# =============================================================================
# Scene Material Collections (from ARNOLD and vMaterials)
# =============================================================================


class SceneMaterialCollections:
    """Curated material collections built from MDL families."""

    TABLE_FAMILIES = ("wood", "stone", "plastic", "ceramic", "metal")
    FLOOR_FAMILIES = ("carpet", "wood", "stone", "concrete", "plastic")
    WALL_FAMILIES = ("architecture", "wall_board", "masonry", "paint", "composite")
    CEILING_FAMILIES = ("architecture", "wall_board", "wood")

    @staticmethod
    def table_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return table/desktop materials sourced from the MDL family registry."""
        return _collect_family_materials(
            families or SceneMaterialCollections.TABLE_FAMILIES,
            max_materials=max_materials,
            warn_missing=warn_missing,
        )

    @staticmethod
    def floor_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return floor materials (carpet/wood/stone/concrete/plastic by default)."""
        return _collect_family_materials(
            families or SceneMaterialCollections.FLOOR_FAMILIES,
            max_materials=max_materials,
            warn_missing=warn_missing,
        )

    @staticmethod
    def wall_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return wall materials (architecture/wall_board/masonry/paint/composite by default)."""
        return _collect_family_materials(
            families or SceneMaterialCollections.WALL_FAMILIES,
            max_materials=max_materials,
            warn_missing=warn_missing,
        )

    @staticmethod
    def ceiling_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return ceiling materials (architecture/wall_board/wood by default)."""
        return _collect_family_materials(
            families or SceneMaterialCollections.CEILING_FAMILIES,
            max_materials=max_materials,
            warn_missing=warn_missing,
        )


def _collect_family_materials(
    families: tuple[str, ...],
    *,
    max_materials: int | None,
    warn_missing: bool,
) -> list[str]:
    """Aggregate unique material paths from the given MDL families."""
    paths: list[str] = []
    for family in families:
        paths.extend(MDLCollections.family(family, warn_missing=warn_missing))

    unique = sorted(dict.fromkeys(paths))
    if max_materials is not None and max_materials > 0 and len(unique) > max_materials:
        unique = unique[:max_materials]
    return unique


# =============================================================================
# Preset Scene Configurations
# =============================================================================


class ScenePresets:
    """Pre-configured scene setups for common scenarios."""

    @staticmethod
    def empty_room(
        room_size: float = 5.0,
        wall_height: float = 3.0,
        wall_thickness: float = 0.1,
        *,
        floor_families: tuple[str, ...] | None = None,
        wall_families: tuple[str, ...] | None = None,
        ceiling_families: tuple[str, ...] | None = None,
    ) -> SceneRandomCfg:
        """Create an empty room with floor, walls, and ceiling.

        Args:
            room_size: Size of the room (square)
            wall_height: Height of walls
            wall_thickness: Thickness of walls
            floor_families: Optional override for floor material families
            wall_families: Optional override for wall material families
            ceiling_families: Optional override for ceiling material families

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
                material_paths=SceneMaterialCollections.floor_materials(families=floor_families),
                selection_strategy="random",
            ),
            wall_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.wall_materials(families=wall_families),
                selection_strategy="random",
            ),
            ceiling_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.ceiling_materials(families=ceiling_families),
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
        *,
        floor_families: tuple[str, ...] | None = None,
        wall_families: tuple[str, ...] | None = None,
        ceiling_families: tuple[str, ...] | None = None,
        table_families: tuple[str, ...] | None = None,
    ) -> SceneRandomCfg:
        """Create a tabletop manipulation workspace.

        Args:
            room_size: Size of the room (square)
            wall_height: Height of walls
            table_size: Size of the table (x, y, z)
            table_height: Height of table surface from ground
            floor_families: Optional override for floor material families
            wall_families: Optional override for wall material families
            ceiling_families: Optional override for ceiling material families
            table_families: Optional override for table material families

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
                material_paths=SceneMaterialCollections.floor_materials(families=floor_families),
                selection_strategy="random",
            ),
            wall_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.wall_materials(families=wall_families),
                selection_strategy="random",
            ),
            ceiling_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.ceiling_materials(families=ceiling_families),
                selection_strategy="random",
            ),
            table_materials=SceneMaterialPoolCfg(
                material_paths=SceneMaterialCollections.table_materials(families=table_families),
                selection_strategy="random",
            ),
            only_if_no_scene=True,
        )

    @staticmethod
    def floor_only(
        floor_size: float = 10.0,
        floor_thickness: float = 0.1,
        *,
        floor_families: tuple[str, ...] | None = None,
    ) -> SceneRandomCfg:
        """Create only a floor (minimal scene).

        Args:
            floor_size: Size of the floor (square)
            floor_thickness: Thickness of floor
            floor_families: Optional override for floor material families

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
                material_paths=SceneMaterialCollections.floor_materials(families=floor_families),
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
        *,
        floor_families: tuple[str, ...] | None = None,
        wall_families: tuple[str, ...] | None = None,
        ceiling_families: tuple[str, ...] | None = None,
        table_families: tuple[str, ...] | None = None,
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
            floor_families: Optional floor material family overrides when floor_materials is None
            wall_families: Optional wall material family overrides when wall_materials is None
            ceiling_families: Optional ceiling material family overrides when ceiling_materials is None
            table_families: Optional table material family overrides when table_materials is None
            only_if_no_scene: Only create if no predefined scene exists

        Returns:
            Scene randomization configuration
        """
        # Use default materials if not provided
        if floor_materials is None:
            floor_materials = SceneMaterialCollections.floor_materials(families=floor_families)
        if wall_materials is None:
            wall_materials = SceneMaterialCollections.wall_materials(families=wall_families)
        if ceiling_materials is None:
            ceiling_materials = SceneMaterialCollections.ceiling_materials(families=ceiling_families)
        if table_materials is None:
            table_materials = SceneMaterialCollections.table_materials(families=table_families)

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
