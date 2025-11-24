"""Material randomization presets and utilities.

Provides common material configurations while allowing full customization. The
material catalogs mirror the assets released at
https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/main/materials.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover - optional dependency in some builds
    HfApi = None
    hf_hub_download = None

from ..material_randomizer import MaterialRandomCfg, MDLMaterialCfg, PBRMaterialCfg, PhysicalMaterialCfg

# =============================================================================
# Material Repository Configuration
# =============================================================================


@dataclass
class MaterialRepository:
    """Configuration for a single MDL material repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "RoboVerseOrg/roboverse_data")
        repo_type: Repository type ("dataset" or "model")
        local_root: Local directory where materials are stored
        remote_root: Root path within the repository (if materials are in subdirectory)
    """

    repo_id: str
    repo_type: str = "dataset"
    local_root: Path | None = None
    remote_root: Path = Path("")

    def __post_init__(self):
        """Set default local_root if not provided."""
        if self.local_root is None:
            repo_name = self.repo_id.split("/")[-1]
            self.local_root = Path(f"roboverse_data/{repo_name}_materials")
        elif isinstance(self.local_root, str):
            self.local_root = Path(self.local_root)

        if isinstance(self.remote_root, str):
            self.remote_root = Path(self.remote_root)


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
    """Collections of MDL material paths organized by type and repository.

    Supports multiple HuggingFace repositories for MDL materials, with automatic
    discovery and optional downloads.

    Example:
        >>> # Use default repository
        >>> wood_materials = MDLCollections.family("wood")
        >>>
        >>> # Register a new repository
        >>> MDLCollections.register_repository(
        ...     name="custom_mdl",
        ...     repo_id="MyOrg/custom_materials",
        ...     local_root="custom_materials",
        ...     remote_root="mdl"
        ... )
        >>>
        >>> # Use materials from specific repository
        >>> custom_wood = MDLCollections.family("wood", repo="custom_mdl")
    """

    REPOSITORIES: dict[str, MaterialRepository] = {
        "default": MaterialRepository(
            repo_id="RoboVerseOrg/roboverse_data",
            local_root=Path("roboverse_data"),
            remote_root=Path("materials"),
        ),
    }

    _HF_API: HfApi | None = None

    @dataclass(frozen=True)
    class FamilyInfo:
        """Metadata about a material family.

        Attributes:
            repo: Repository name (must exist in REPOSITORIES)
            path: Relative path to material directory or file
            description: Optional human-readable description
        """

        repo: str
        path: str
        description: str | None = None

        def slug(self) -> str:
            """Return a canonical 'repo:path' identifier."""
            return f"{self.repo}:{self.path}"

    FAMILY_REGISTRY: dict[str, tuple[FamilyInfo, ...]] = {
        "wood": (
            FamilyInfo("default", "arnold/Wood", "General-purpose wood grains (Arnold)"),
            FamilyInfo("default", "vMaterials_2/Wood", "Extended wood library (vMaterials 2)"),
        ),
        "architecture": (FamilyInfo("default", "arnold/Architecture", "Ceiling/roof/shingle surfaces"),),
        "carpet": (
            FamilyInfo("default", "arnold/Carpet", "Carpet and soft fabrics (Arnold)"),
            FamilyInfo("default", "vMaterials_2/Carpet", "Carpet collection (vMaterials 2)"),
        ),
        "masonry": (
            FamilyInfo("default", "arnold/Masonry", "Bricks and masonry blocks (Arnold)"),
            FamilyInfo("default", "vMaterials_2/Masonry", "Bricks and stonework (vMaterials 2)"),
        ),
        "wall_board": (FamilyInfo("default", "arnold/Wall_Board", "Wall boards and trims"),),
        "water": (
            FamilyInfo("default", "arnold/Natural/Water", "Water materials (subdirectory)"),
            FamilyInfo("default", "arnold/Water_Opaque.mdl", "Opaque water shader (single file)"),
        ),
        "metal": (FamilyInfo("default", "vMaterials_2/Metal", "Brushed/polished metal set"),),
        "stone": (FamilyInfo("default", "vMaterials_2/Stone", "Stone, terrazzo, rock surfaces"),),
        "plastic": (FamilyInfo("default", "vMaterials_2/Plastic", "Plastics and polymers"),),
        "fabric": (FamilyInfo("default", "vMaterials_2/Fabric", "Textiles and cloth surfaces"),),
        "leather": (FamilyInfo("default", "vMaterials_2/Leather", "Leather, suede, skin"),),
        "glass": (FamilyInfo("default", "vMaterials_2/Glass", "Glass and translucent materials"),),
        "ceramic": (FamilyInfo("default", "vMaterials_2/Ceramic", "Ceramic and tiles"),),
        "concrete": (FamilyInfo("default", "vMaterials_2/Concrete", "Concrete, cement, rough surfaces"),),
        "paper": (FamilyInfo("default", "vMaterials_2/Paper", "Paper and cardboard"),),
        "paint": (FamilyInfo("default", "vMaterials_2/Paint", "Coated paint finishes"),),
        "ground": (FamilyInfo("default", "vMaterials_2/Ground", "Soil, sand, and outdoor ground"),),
        "gems": (FamilyInfo("default", "vMaterials_2/Gems", "Gemstones"),),
        "composite": (FamilyInfo("default", "vMaterials_2/Composite", "Composite technical materials"),),
        "plaster": (FamilyInfo("default", "vMaterials_2/Plaster", "Plaster and stucco"),),
        "liquids": (FamilyInfo("default", "vMaterials_2/Liquids", "Various liquids"),),
        "natural": (FamilyInfo("default", "arnold/Natural", "Natural materials (Arnold)"),),
        "templates": (FamilyInfo("default", "arnold/Templates", "Material templates (Arnold)"),),
        "other": (FamilyInfo("default", "vMaterials_2/Other", "Miscellaneous utility shaders"),),
    }

    @classmethod
    def register_repository(
        cls,
        name: str,
        repo_id: str,
        repo_type: str = "dataset",
        local_root: str | Path | None = None,
        remote_root: str | Path = "",
    ) -> None:
        """Register a new MDL material repository.

        Args:
            name: Short name for the repository
            repo_id: HuggingFace repository ID
            repo_type: Repository type
            local_root: Local directory for materials
            remote_root: Root path within the repository
        """
        cls.REPOSITORIES[name] = MaterialRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            local_root=Path(local_root) if local_root else None,
            remote_root=Path(remote_root),
        )

    @classmethod
    def family(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        repo: str | None = None,
        max_materials: int | None = None,
        warn_missing: bool = True,
        use_remote_manifest: bool = True,
    ) -> list[str]:
        """Get materials from a family (wood, metal, plastic, etc.).

        Returns paths from HuggingFace manifest (if available) or local scan (fallback).
        Actual download happens on-demand when materials are used.

        Args:
            name: Family name (e.g., 'wood', 'metal', 'plastic')
            use_remote_manifest: If True (default), query HuggingFace for complete list.
                                If False, only scan local directory.
            root: Optional custom root path (overrides repository configuration)
            repo: Optional repository name to filter by (if None, uses all repositories)
            max_materials: Optional limit on number of materials returned
            warn_missing: Whether to warn about missing materials

        Returns:
            List of MDL file paths from all matching family entries
        """
        key = name.lower()
        infos = cls.FAMILY_REGISTRY.get(key)
        if not infos:
            known = ", ".join(sorted(cls.FAMILY_REGISTRY))
            raise KeyError(f"Unknown material family '{name}'. Available families: {known}.")

        collected: list[str] = []
        for info in infos:
            # Filter by repository if specified
            if repo is not None and info.repo != repo:
                continue

            # Determine the repository configuration
            repo_config = cls.REPOSITORIES.get(info.repo)
            if not repo_config:
                if warn_missing:
                    import warnings

                    warnings.warn(f"Repository '{info.repo}' not found, skipping family '{name}'", stacklevel=3)
                continue

            # Determine base path
            if root is not None:
                base_root = Path(root)
            else:
                # Combine local_root with remote_root to form full local path
                # E.g., local_root="roboverse_data" + remote_root="materials" -> "roboverse_data/materials"
                base_root = repo_config.local_root / repo_config.remote_root

            # Check if path is a single file or directory
            target_path = base_root / info.path
            if target_path.suffix == ".mdl":
                # Single MDL file
                paths = [target_path]
            else:
                # Directory
                paths = [target_path]

            collected.extend(
                cls._collect_from_paths(
                    paths, warn_missing=warn_missing, repo=info.repo, use_remote_manifest=use_remote_manifest
                )
            )

        # Deduplicate and sort
        unique = sorted(dict.fromkeys(collected))

        # Apply limit if specified
        if max_materials is not None and len(unique) > max_materials:
            unique = unique[:max_materials]

        return unique

    @classmethod
    def families(cls) -> dict[str, tuple[MDLCollections.FamilyInfo, ...]]:
        """Expose the family registry (copy) for UI/debug use."""
        return {name: tuple(infos) for name, infos in cls.FAMILY_REGISTRY.items()}

    @classmethod
    def families_materials(
        cls,
        families: Sequence[str],
        *,
        root: str | Path | None = None,
        repo: str | None = None,
        warn_missing: bool = True,
    ) -> list[str]:
        """Collect merged material lists from multiple families (deduplicated + sorted).

        Args:
            families: List of family names to collect
            root: Optional custom root path
            repo: Optional repository name
            warn_missing: Whether to warn about missing materials

        Returns:
            Merged list of MDL file paths
        """
        paths: list[str] = []
        for family in families:
            paths.extend(cls.family(family, root=root, repo=repo, warn_missing=warn_missing))

        return sorted(dict.fromkeys(paths))

    @classmethod
    def available_families(cls) -> list[str]:
        """Get list of all available family names.

        Returns:
            Sorted list of family names
        """
        return sorted(cls.FAMILY_REGISTRY.keys())

    @classmethod
    def _collect_from_paths(
        cls,
        paths: Iterable[Path],
        *,
        warn_missing: bool = True,
        repo: str | None = None,
        use_remote_manifest: bool = True,
    ) -> list[str]:
        """Collect ``.mdl`` files under the provided directories.

        Args:
            paths: Directory or file paths to search
            warn_missing: Whether to warn about missing materials
            repo: Optional repository name for remote manifest lookups
            use_remote_manifest: If True, query HuggingFace for complete list; if False, scan local only

        Returns:
            List of MDL file paths
        """
        mdl_paths: list[str] = []
        missing: list[str] = []

        for target in paths:
            # Check if user wants remote manifest
            if use_remote_manifest:
                # Try remote first (default: get complete list from HuggingFace)
                remote_paths = cls._collect_remote_mdl_paths(target, repo=repo)

                if remote_paths:
                    # Use remote manifest (complete list)
                    mdl_paths.extend(remote_paths)
                    continue  # Skip local scan

            # Local-only mode or remote unavailable
            if target.is_dir():
                # Local directory scan
                mdl_paths.extend(sorted(p.as_posix() for p in target.rglob("*.mdl")))
            elif target.is_file() and target.suffix.lower() == ".mdl":
                # Single file
                mdl_paths.append(target.as_posix())
            else:
                # Not found locally and remote disabled/unavailable
                missing.append(target.as_posix())

        if missing and warn_missing:
            repo_info = ""
            if repo and repo in cls.REPOSITORIES:
                repo_config = cls.REPOSITORIES[repo]
                repo_info = f" from repository '{repo}' ({repo_config.repo_id})"
            warning_msg = f"Missing material assets{repo_info}:\n  - " + "\n  - ".join(missing)
            warnings.warn(warning_msg, stacklevel=2)

        # Remove duplicates before returning a deterministic, sorted list.
        unique = list(dict.fromkeys(mdl_paths))
        return sorted(unique)

    @classmethod
    def _collect_remote_mdl_paths(cls, target: Path, repo: str | None = None) -> list[str]:
        """Return remote ``.mdl`` paths that should exist under ``target``.

        Converts remote HuggingFace paths back into the expected local layout so
        downstream code can keep using ``roboverse_data/...`` style strings.

        Args:
            target: Local target path to search for
            repo: Optional repository name (defaults to 'default')

        Returns:
            List of local MDL paths that should exist based on remote manifest
        """
        repo_name = repo or "default"
        if repo_name not in cls.REPOSITORIES:
            return []

        repo_config = cls.REPOSITORIES[repo_name]
        manifest = cls._remote_manifest(repo_name)
        if not manifest:
            return []

        try:
            rel = target.relative_to(repo_config.local_root)
        except ValueError:
            # Target path doesn't match this repository
            return []

        # rel already contains remote_root (e.g., "materials/arnold/Wood")
        # So we should use rel directly as the remote prefix, not combine with remote_root again
        remote_prefix = rel.as_posix()
        normalized_prefix = remote_prefix.rstrip("/")

        if normalized_prefix.endswith(".mdl"):
            candidates = [remote_prefix] if remote_prefix in manifest else []
        else:
            prefix = normalized_prefix + "/"
            candidates = [path for path in manifest if path.startswith(prefix) and path.endswith(".mdl")]
            # Sort candidates to ensure deterministic order for reproducibility
            candidates = sorted(candidates)

        collected: list[str] = []
        for remote_path in candidates:
            # Convert remote path back to local path
            # remote_path is like "materials/arnold/Wood/Ash.mdl"
            # We want "roboverse_data/materials/arnold/Wood/Ash.mdl"
            collected.append((repo_config.local_root / remote_path).as_posix())

        return collected

    @classmethod
    @lru_cache(maxsize=8)
    def _remote_manifest(cls, repo: str = "default") -> tuple[str, ...]:
        """Fetch the list of files hosted on HuggingFace (cached).

        Args:
            repo: Repository name (defaults to 'default')

        Returns:
            Tuple of file paths in the repository
        """
        if repo not in cls.REPOSITORIES:
            return ()

        repo_config = cls.REPOSITORIES[repo]
        api = cls._get_hf_api()
        if api is None:
            return ()

        try:
            files = api.list_repo_files(repo_id=repo_config.repo_id, repo_type=repo_config.repo_type)
            # Sort files to ensure deterministic order for reproducibility
            files = sorted(files)
        except Exception as exc:  # pragma: no cover - network/SDK issues
            warnings.warn(
                f"Failed to query HuggingFace repo '{repo_config.repo_id}': {exc}",
                stacklevel=2,
            )
            return ()

        return tuple(files)

    @classmethod
    def _get_hf_api(cls) -> HfApi | None:
        if HfApi is None:
            return None
        if cls._HF_API is None:
            cls._HF_API = HfApi()
        return cls._HF_API


# =============================================================================
# Preset Material Configurations
# =============================================================================


def _phys_factory(friction_range: tuple[float, float], restitution_range: tuple[float, float]):
    return lambda: PhysicalMaterialCfg(
        friction_range=friction_range,
        restitution_range=restitution_range,
        enabled=True,
    )


def _pbr_factory(
    roughness: tuple[float, float],
    metallic: tuple[float, float],
    color: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
):
    return lambda: PBRMaterialCfg(
        roughness_range=roughness,
        metallic_range=metallic,
        diffuse_color_range=color,
        enabled=True,
    )


class MaterialPresets:
    """Pre-configured material setups for common scenarios."""

    _FAMILY_PHYSICAL_FACTORIES: dict[str, Callable[[], PhysicalMaterialCfg]] = {
        "metal": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_LOW),
        "wood": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "stone": _phys_factory(MaterialProperties.FRICTION_HIGH, MaterialProperties.RESTITUTION_LOW),
        "plastic": _phys_factory(MaterialProperties.FRICTION_LOW, MaterialProperties.RESTITUTION_MEDIUM),
        "fabric": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_HIGH),
        "carpet": _phys_factory(MaterialProperties.FRICTION_HIGH, MaterialProperties.RESTITUTION_MEDIUM),
        "leather": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "glass": _phys_factory(MaterialProperties.FRICTION_LOW, MaterialProperties.RESTITUTION_LOW),
        "ceramic": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "concrete": _phys_factory(MaterialProperties.FRICTION_HIGH, MaterialProperties.RESTITUTION_LOW),
        "paper": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "paint": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "ground": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_LOW),
        "water": _phys_factory(MaterialProperties.FRICTION_LOW, MaterialProperties.RESTITUTION_HIGH),
        "wall_board": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_LOW),
        "architecture": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_LOW),
        "masonry": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_LOW),
        "gems": _phys_factory(MaterialProperties.FRICTION_LOW, MaterialProperties.RESTITUTION_MEDIUM),
        "composite": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
        "other": _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM),
    }

    _GENERIC_PHYSICAL_FACTORY = _phys_factory(MaterialProperties.FRICTION_MEDIUM, MaterialProperties.RESTITUTION_MEDIUM)

    _FAMILY_PBR_FACTORIES: dict[str, Callable[[], PBRMaterialCfg]] = {
        "metal": _pbr_factory(
            MaterialProperties.ROUGHNESS_SMOOTH,
            MaterialProperties.METALLIC_FULL,
            ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
        ),
        "wood": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            ((0.3, 0.8), (0.2, 0.6), (0.1, 0.4)),
        ),
        "plastic": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_BRIGHT,
        ),
        "fabric": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "carpet": _pbr_factory(
            MaterialProperties.ROUGHNESS_ROUGH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "leather": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_WARM,
        ),
        "glass": _pbr_factory(
            MaterialProperties.ROUGHNESS_SMOOTH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_COOL,
        ),
        "ceramic": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "stone": _pbr_factory(
            MaterialProperties.ROUGHNESS_ROUGH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "concrete": _pbr_factory(
            MaterialProperties.ROUGHNESS_ROUGH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "paper": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "paint": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_FULL,
        ),
        "ground": _pbr_factory(
            MaterialProperties.ROUGHNESS_ROUGH,
            MaterialProperties.METALLIC_NON,
            ((0.2, 0.6), (0.3, 0.7), (0.1, 0.5)),
        ),
        "water": _pbr_factory(
            MaterialProperties.ROUGHNESS_SMOOTH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_COOL,
        ),
        "gems": _pbr_factory(
            MaterialProperties.ROUGHNESS_SMOOTH,
            MaterialProperties.METALLIC_FULL,
            MaterialProperties.COLOR_BRIGHT,
        ),
        "architecture": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_PARTIAL,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "masonry": _pbr_factory(
            MaterialProperties.ROUGHNESS_ROUGH,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "wall_board": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_NON,
            MaterialProperties.COLOR_NEUTRAL,
        ),
        "composite": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_PARTIAL,
            MaterialProperties.COLOR_FULL,
        ),
        "other": _pbr_factory(
            MaterialProperties.ROUGHNESS_MEDIUM,
            MaterialProperties.METALLIC_PARTIAL,
            MaterialProperties.COLOR_FULL,
        ),
    }

    _GENERIC_PBR_FACTORY = _pbr_factory(
        MaterialProperties.ROUGHNESS_MEDIUM,
        MaterialProperties.METALLIC_PARTIAL,
        MaterialProperties.COLOR_NEUTRAL,
    )

    @staticmethod
    def plastic_object(obj_name: str, color_range: tuple = MaterialProperties.COLOR_BRIGHT) -> MaterialRandomCfg:
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
        )

    @staticmethod
    def rubber_object(obj_name: str, color_range: tuple = MaterialProperties.COLOR_NEUTRAL) -> MaterialRandomCfg:
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
        )

    @staticmethod
    def mdl_family_object(
        obj_name: str,
        family: str | Sequence[str],
        *,
        use_mdl: bool = True,
        assets_root: str | Path | None = None,
        mdl_paths: list[str] | None = None,
        physical_config: PhysicalMaterialCfg | None = None,
        fallback_pbr: PBRMaterialCfg | None = None,
        warn_missing_assets: bool = True,
    ) -> MaterialRandomCfg:
        """Create an object preset driven by one or more MDL families (wood, metal, etc.)."""
        families = _ensure_family_tuple(family)
        primary_family = families[0]

        physical = physical_config or MaterialPresets._family_physical_default(primary_family)
        config = MaterialRandomCfg(obj_name=obj_name, physical=physical)

        if use_mdl:
            resolved_paths = mdl_paths
            if resolved_paths is None:
                resolved_paths = MDLCollections.families_materials(
                    families, root=assets_root, warn_missing=warn_missing_assets
                )

            if resolved_paths:
                config.mdl = MDLMaterialCfg(mdl_paths=sorted(dict.fromkeys(resolved_paths)), enabled=True)
            elif warn_missing_assets:
                warnings.warn(
                    f"No MDL assets found for families {families}. Falling back to PBR if provided.",
                    stacklevel=2,
                )

        if not getattr(config, "mdl", None):
            pbr_cfg = fallback_pbr or MaterialPresets._family_pbr_default(primary_family)
            if pbr_cfg:
                config.pbr = pbr_cfg

        return config

    @staticmethod
    def metal_object(
        obj_name: str,
        use_mdl: bool = True,
        mdl_base_path: str = "roboverse_data/materials/vMaterials_2/Metal",
    ) -> MaterialRandomCfg:
        """Deprecated metal preset wrapper kept for backward compatibility."""
        warnings.warn(
            "MaterialPresets.metal_object is deprecated; use MaterialPresets.mdl_family_object(..., family='metal').",
            DeprecationWarning,
            stacklevel=2,
        )

        mdl_paths = None
        default_path = "roboverse_data/materials/vMaterials_2/Metal"
        if mdl_base_path != default_path:
            mdl_paths = MDLCollections._collect_from_paths([Path(mdl_base_path)])

        return MaterialPresets.mdl_family_object(
            obj_name=obj_name,
            family="metal",
            use_mdl=use_mdl,
            mdl_paths=mdl_paths,
        )

    @staticmethod
    def wood_object(
        obj_name: str,
        use_mdl: bool = True,
        mdl_base_path: str = "roboverse_data/materials/arnold/Wood",
    ) -> MaterialRandomCfg:
        """Deprecated wood preset wrapper kept for backward compatibility."""
        warnings.warn(
            "MaterialPresets.wood_object is deprecated; use MaterialPresets.mdl_family_object(..., family='wood').",
            DeprecationWarning,
            stacklevel=2,
        )

        mdl_paths = None
        default_path = "roboverse_data/materials/arnold/Wood"
        if mdl_base_path != default_path:
            mdl_paths = MDLCollections._collect_from_paths([Path(mdl_base_path)])

        return MaterialPresets.mdl_family_object(
            obj_name=obj_name,
            family="wood",
            use_mdl=use_mdl,
            mdl_paths=mdl_paths,
        )

    @staticmethod
    def custom_object(
        obj_name: str,
        physical_config: PhysicalMaterialCfg | None = None,
        pbr_config: PBRMaterialCfg | None = None,
        mdl_config: MDLMaterialCfg | None = None,
    ) -> MaterialRandomCfg:
        """Create fully customizable material configuration."""
        return MaterialRandomCfg(
            obj_name=obj_name,
            physical=physical_config,
            pbr=pbr_config,
            mdl=mdl_config,
        )

    @classmethod
    def _family_physical_default(cls, family: str) -> PhysicalMaterialCfg | None:
        factory = cls._FAMILY_PHYSICAL_FACTORIES.get(family, cls._GENERIC_PHYSICAL_FACTORY)
        return factory() if factory else None

    @classmethod
    def _family_pbr_default(cls, family: str) -> PBRMaterialCfg | None:
        factory = cls._FAMILY_PBR_FACTORIES.get(family, cls._GENERIC_PBR_FACTORY)
        return factory() if factory else None

    @classmethod
    def families_materials(
        cls,
        families: Sequence[str],
        *,
        root: str | Path | None = None,
        repo: str | None = None,
        warn_missing: bool = True,
    ) -> list[str]:
        """Collect merged material lists from multiple families (deduplicated + sorted).

        Args:
            families: List of family names to collect
            root: Optional custom root path
            repo: Optional repository name
            warn_missing: Whether to warn about missing materials

        Returns:
            Merged list of MDL file paths
        """
        paths: list[str] = []
        for family in families:
            paths.extend(MDLCollections.family(family, root=root, repo=repo, warn_missing=warn_missing))

        return sorted(dict.fromkeys(paths))


def _ensure_family_tuple(family: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(family, str):
        return (family,)
    if isinstance(family, tuple):
        return family
    return tuple(family)
