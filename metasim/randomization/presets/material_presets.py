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
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency in some builds
    HfApi = None

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
    """Collections of MDL material paths organized by type.

    The collections below mirror the dataset layout hosted on Hugging Face so that
    new assets are picked up automatically once downloaded locally. Call
    :meth:`materials` for any dataset/category pair or use the higher-level
    helpers for commonly used material families.
    """

    HUGGINGFACE_MATERIALS_URL = "https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/main/materials"
    DEFAULT_ROOT = Path("roboverse_data/materials")
    HF_REPO_ID = "RoboVerseOrg/roboverse_data"
    HF_REPO_TYPE = "dataset"
    HF_REMOTE_ROOT = Path("materials")
    _HF_API: HfApi | None = None

    ARNOLD_COLLECTIONS = {
        "architecture": (Path("arnold/Architecture"),),
        "carpet": (Path("arnold/Carpet"),),
        "masonry": (Path("arnold/Masonry"),),
        "natural": (Path("arnold/Natural"),),
        "templates": (Path("arnold/Templates"),),
        "wall_board": (Path("arnold/Wall_Board"),),
        "wood": (Path("arnold/Wood"),),
        "water": (Path("arnold/Natural/Water"), Path("arnold/Water_Opaque.mdl")),
    }

    VMATERIALS_COLLECTIONS = {
        "carpet": (Path("vMaterials_2/Carpet"),),
        "ceramic": (Path("vMaterials_2/Ceramic"),),
        "composite": (Path("vMaterials_2/Composite"),),
        "concrete": (Path("vMaterials_2/Concrete"),),
        "fabric": (Path("vMaterials_2/Fabric"),),
        "gems": (Path("vMaterials_2/Gems"),),
        "glass": (Path("vMaterials_2/Glass"),),
        "ground": (Path("vMaterials_2/Ground"),),
        "leather": (Path("vMaterials_2/Leather"),),
        "liquids": (Path("vMaterials_2/Liquids"),),
        "masonry": (Path("vMaterials_2/Masonry"),),
        "metal": (Path("vMaterials_2/Metal"),),
        "other": (Path("vMaterials_2/Other"),),
        "paint": (Path("vMaterials_2/Paint"),),
        "paper": (Path("vMaterials_2/Paper"),),
        "plaster": (Path("vMaterials_2/Plaster"),),
        "plastic": (Path("vMaterials_2/Plastic"),),
        "stone": (Path("vMaterials_2/Stone"),),
        "wood": (Path("vMaterials_2/Wood"),),
    }

    _DATASET_MAP = {
        "arnold": ARNOLD_COLLECTIONS,
        "vmaterials_2": VMATERIALS_COLLECTIONS,
    }

    _ALIASES = {
        "arnold": "arnold",
        "vmaterials_2": "vmaterials_2",
        "vmaterials": "vmaterials_2",
        "vmaterials2": "vmaterials_2",
        "v-materials": "vmaterials_2",
    }

    @dataclass(frozen=True)
    class FamilyInfo:
        """Metadata about a material family within a dataset."""

        dataset: str
        category: str
        description: str | None = None

        def slug(self) -> str:
            """Return a canonical ``dataset:category`` identifier."""
            return f"{self.dataset}:{self.category}"

    FAMILY_REGISTRY: dict[str, tuple[MDLCollections.FamilyInfo, ...]] = {
        # Arnold defaults
        "wood": (
            FamilyInfo("arnold", "wood", "General-purpose wood grains (Arnold)"),
            FamilyInfo("vmaterials_2", "wood", "Extended wood library (vMaterials2)"),
        ),
        "architecture": (FamilyInfo("arnold", "architecture", "Ceiling/roof/shingle surfaces"),),
        "carpet": (
            FamilyInfo("arnold", "carpet", "Carpet and soft fabrics (Arnold)"),
            FamilyInfo("vmaterials_2", "carpet", "Carpet collection (vMaterials2)"),
        ),
        "masonry": (
            FamilyInfo("arnold", "masonry", "Bricks and masonry blocks (Arnold)"),
            FamilyInfo("vmaterials_2", "masonry", "Bricks and stonework (vMaterials2)"),
        ),
        "wall_board": (FamilyInfo("arnold", "wall_board", "Wall boards and trims"),),
        "water": (FamilyInfo("arnold", "water", "Opaque + clear water shaders"),),
        # NVIDIA vMaterials v2 defaults
        "metal": (FamilyInfo("vmaterials_2", "metal", "Brushed/polished metal set"),),
        "stone": (FamilyInfo("vmaterials_2", "stone", "Stone, terrazzo, rock surfaces"),),
        "plastic": (FamilyInfo("vmaterials_2", "plastic", "Plastics and polymers"),),
        "fabric": (FamilyInfo("vmaterials_2", "fabric", "Textiles and cloth surfaces"),),
        "leather": (FamilyInfo("vmaterials_2", "leather", "Leather, suede, skin"),),
        "glass": (FamilyInfo("vmaterials_2", "glass", "Glass and translucent materials"),),
        "ceramic": (FamilyInfo("vmaterials_2", "ceramic", "Ceramic and tiles"),),
        "concrete": (FamilyInfo("vmaterials_2", "concrete", "Concrete, cement, rough surfaces"),),
        "paper": (FamilyInfo("vmaterials_2", "paper", "Paper and cardboard"),),
        "paint": (FamilyInfo("vmaterials_2", "paint", "Coated paint finishes"),),
        "ground": (FamilyInfo("vmaterials_2", "ground", "Soil, sand, and outdoor ground"),),
        "gems": (FamilyInfo("vmaterials_2", "gems", "Gemstones"),),
        "composite": (FamilyInfo("vmaterials_2", "composite", "Composite technical materials"),),
        "other": (FamilyInfo("vmaterials_2", "other", "Miscellaneous utility shaders"),),
    }

    @classmethod
    def family(cls, name: str, *, root: str | Path | None = None, warn_missing: bool = True) -> list[str]:
        """Get an easy-to-remember *family* (wood, metal, plastic, ...)."""
        base_root = Path(root) if root else cls.DEFAULT_ROOT
        key = name.lower()
        infos = cls.FAMILY_REGISTRY.get(key)
        if not infos:
            known = ", ".join(sorted(cls.FAMILY_REGISTRY))
            raise KeyError(f"Unknown material family '{name}'. Available families: {known}.")

        collected: list[str] = []
        for info in infos:
            collected.extend(cls.materials(info.dataset, info.category, root=base_root, warn_missing=warn_missing))
        # Deduplicate and sort to keep reproducible order when multiple datasets contribute.
        return sorted(dict.fromkeys(collected))

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
        warn_missing: bool = True,
    ) -> list[str]:
        """Collect merged material lists from multiple families (deduplicated + sorted)."""
        paths: list[str] = []
        for family in families:
            paths.extend(cls.family(family, root=root, warn_missing=warn_missing))

        return sorted(dict.fromkeys(paths))

    @classmethod
    def materials(
        cls, dataset: str, category: str, root: str | Path | None = None, *, warn_missing: bool = True
    ) -> list[str]:
        """Return every ``.mdl`` inside the requested dataset/category grouping."""
        dataset_key = cls._normalize_dataset(dataset)
        category_key = category.lower()

        if dataset_key not in cls._DATASET_MAP:
            raise KeyError(f"Unknown MDL dataset '{dataset}'. Known datasets: {tuple(cls._DATASET_MAP.keys())}.")

        registry = cls._DATASET_MAP[dataset_key]
        if category_key not in registry:
            known = ", ".join(sorted(registry.keys()))
            raise KeyError(f"Unknown category '{category}' for dataset '{dataset_key}'. Available: {known}.")

        base_root = Path(root) if root else cls.DEFAULT_ROOT
        targets = [base_root / rel for rel in registry[category_key]]
        return cls._collect_from_paths(targets, warn_missing=warn_missing)

    @classmethod
    def catalog(cls, root: str | Path | None = None, *, warn_missing: bool = False) -> dict[str, dict[str, list[str]]]:
        """Build a full catalog ``{dataset: {category: [paths]}}`` for quick inspection."""
        catalog: dict[str, dict[str, list[str]]] = {}
        for dataset, categories in cls._DATASET_MAP.items():
            catalog[dataset] = {}
            for category in categories:
                catalog[dataset][category] = cls.materials(dataset, category, root=root, warn_missing=warn_missing)
        return catalog

    @classmethod
    def available_categories(cls) -> dict[str, tuple[str, ...]]:
        """Expose supported Hugging Face groupings for discoverability."""
        return {dataset: tuple(sorted(categories.keys())) for dataset, categories in cls._DATASET_MAP.items()}

    @classmethod
    def _collect_from_paths(cls, paths: Iterable[Path], *, warn_missing: bool = True) -> list[str]:
        """Collect ``.mdl`` files under the provided directories."""
        mdl_paths: list[str] = []
        missing: list[str] = []

        for target in paths:
            if target.is_dir():
                # Sort to ensure deterministic order for reproducibility
                mdl_paths.extend(sorted(p.as_posix() for p in target.rglob("*.mdl")))
            elif target.is_file() and target.suffix.lower() == ".mdl":
                mdl_paths.append(target.as_posix())
            else:
                remote_paths = cls._collect_remote_mdl_paths(target)
                if remote_paths:
                    mdl_paths.extend(remote_paths)
                else:
                    missing.append(target.as_posix())

        if missing and warn_missing:
            warning_msg = (
                "Missing material assets:\n  - "
                + "\n  - ".join(missing)
                + f"\nDownload them from {MDLCollections.HUGGINGFACE_MATERIALS_URL}."
            )
            warnings.warn(warning_msg, stacklevel=2)

        # Remove duplicates before returning a deterministic, sorted list.
        unique = list(dict.fromkeys(mdl_paths))
        return sorted(unique)

    @classmethod
    def _collect_remote_mdl_paths(cls, target: Path) -> list[str]:
        """Return remote ``.mdl`` paths that should exist under ``target``.

        Converts remote HuggingFace paths back into the expected local layout so
        downstream code can keep using ``roboverse_data/...`` style strings.
        """
        manifest = cls._remote_manifest()
        if not manifest:
            return []

        try:
            rel = target.relative_to(cls.DEFAULT_ROOT)
        except ValueError:
            # Custom roots can't rely on the shared HuggingFace dataset layout.
            return []

        remote_prefix = (cls.HF_REMOTE_ROOT / rel).as_posix()
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
            try:
                relative_remote = Path(remote_path).relative_to(cls.HF_REMOTE_ROOT)
            except ValueError:
                continue
            collected.append((cls.DEFAULT_ROOT / relative_remote).as_posix())

        return collected

    @classmethod
    @lru_cache(maxsize=1)
    def _remote_manifest(cls) -> tuple[str, ...]:
        """Fetch the list of files hosted on HuggingFace (cached)."""
        api = cls._get_hf_api()
        if api is None:
            return ()

        try:
            files = api.list_repo_files(repo_id=cls.HF_REPO_ID, repo_type=cls.HF_REPO_TYPE)
            # Sort files to ensure deterministic order for reproducibility
            files = sorted(files)
        except Exception as exc:  # pragma: no cover - network/SDK issues
            warnings.warn(
                f"Failed to query HuggingFace repo '{cls.HF_REPO_ID}': {exc}",
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

    @classmethod
    def _normalize_dataset(cls, dataset: str) -> str:
        key = dataset.lower()
        if key in cls._ALIASES:
            return cls._ALIASES[key]
        return key


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
    def mdl_family_object(
        obj_name: str,
        family: str | Sequence[str],
        *,
        randomization_mode: str = "combined",
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
        config = MaterialRandomCfg(obj_name=obj_name, physical=physical, randomization_mode=randomization_mode)

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
        randomization_mode: str = "combined",
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
            randomization_mode=randomization_mode,
            use_mdl=use_mdl,
            mdl_paths=mdl_paths,
        )

    @staticmethod
    def wood_object(
        obj_name: str,
        use_mdl: bool = True,
        mdl_base_path: str = "roboverse_data/materials/arnold/Wood",
        randomization_mode: str = "combined",
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
            randomization_mode=randomization_mode,
            use_mdl=use_mdl,
            mdl_paths=mdl_paths,
        )

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
        warn_missing: bool = True,
    ) -> list[str]:
        """Collect merged material lists from multiple families (deduplicated + sorted)."""
        paths: list[str] = []
        for family in families:
            paths.extend(cls.family(family, root=root, warn_missing=warn_missing))

        return sorted(dict.fromkeys(paths))


def _ensure_family_tuple(family: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(family, str):
        return (family,)
    if isinstance(family, tuple):
        return family
    return tuple(family)
