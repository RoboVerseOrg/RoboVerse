"""Preset configurations for scene randomization.

This module provides curated material collections and preset scene configurations
for the 3-layer hierarchical scene randomization system:
- Layer 0 (Environment): Backgrounds, rooms, walls, floors, ceilings
- Layer 1 (Workspace): Tables, desktops, manipulation surfaces
- Layer 2 (Objects): Static distractor objects
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

from metasim.randomization.scene_randomizer import (
    EnvironmentLayerCfg,
    ManualGeometryCfg,
    ObjectsLayerCfg,
    SceneRandomCfg,
    USDAssetCfg,
    USDAssetPoolCfg,
    WorkspaceLayerCfg,
)

from .material_presets import MDLCollections

# =============================================================================
# Repository Configuration for USD Assets
# =============================================================================


@dataclass
class AssetRepository:
    """Configuration for a single asset repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "HorizonRobotics/EmbodiedGenData")
        repo_type: Repository type ("dataset" or "model")
        local_root: Local directory where assets are stored
        remote_root: Root path within the repository (if assets are in subdirectory)
    """

    repo_id: str
    repo_type: str = "dataset"
    local_root: Path | None = None
    remote_root: Path = Path("")

    def __post_init__(self):
        """Set default local_root if not provided."""
        if self.local_root is None:
            repo_name = self.repo_id.split("/")[-1]
            self.local_root = Path(f"roboverse_data/assets/{repo_name}")
        elif isinstance(self.local_root, str):
            self.local_root = Path(self.local_root)

        if isinstance(self.remote_root, str):
            self.remote_root = Path(self.remote_root)


# =============================================================================
# USD Asset Collections
# =============================================================================


class USDCollections:
    """USD/URDF asset collections organized by type and repository.

    Provides organized access to USD and URDF assets from multiple HuggingFace repositories,
    supporting both local file access and automatic download. This is the base
    collection class, similar to MDLCollections for materials.

    Supported formats:
        - USD: .usd, .usda, .usdz (native IsaacSim format)
        - URDF: .urdf (automatically converted to USD at runtime by IsaacSim)

    Example:
        >>> # Get furniture from a family (returns USD or URDF paths)
        >>> chairs = USDCollections.family("chair", repo="embodiedgen")
        >>>
        >>> # Get all tables from the table family
        >>> tables = USDCollections.family("table")  # Returns URDF paths from EmbodiedGen
        >>>
        >>> # Register a new repository
        >>> USDCollections.register_repository(
        ...     name="custom_usd",
        ...     repo_id="MyOrg/custom_assets",
        ...     local_root="custom_assets"
        ... )
    """

    REPOSITORIES: dict[str, AssetRepository] = {
        "embodiedgen": AssetRepository(
            repo_id="HorizonRobotics/EmbodiedGenData",
            local_root=Path("EmbodiedGenData"),
            remote_root=Path(""),
        ),
        "roboverse": AssetRepository(
            repo_id="RoboVerseOrg/roboverse_data",
            local_root=Path("roboverse_data"),
            remote_root=Path(""),
        ),
    }

    @dataclass(frozen=True)
    class FamilyInfo:
        """Metadata about a USD asset family.

        Attributes:
            repo: Repository name (must exist in REPOSITORIES)
            path: Relative path to USD asset directory
            description: Optional human-readable description
        """

        repo: str
        path: str
        description: str | None = None

        def slug(self) -> str:
            """Return a canonical 'repo:path' identifier."""
            return f"{self.repo}:{self.path}"

    FAMILY_REGISTRY: dict[str, tuple[FamilyInfo, ...]] = {
        "table": (FamilyInfo("embodiedgen", "dataset/basic_furniture/table", "Tables from EmbodiedGen"),),
        "chair": (FamilyInfo("embodiedgen", "dataset/basic_furniture/chair", "Chairs from EmbodiedGen"),),
        "kujiale": (FamilyInfo("roboverse", "scenes/kujiale", "Kujiale interior scenes"),),
        "decorations": (
            FamilyInfo("embodiedgen", "dataset/desktop_supplies/decorations", "Decorations from EmbodiedGen"),
        ),
        "office_stationery": (
            FamilyInfo(
                "embodiedgen", "dataset/desktop_supplies/office_stationery", "Office stationery from EmbodiedGen"
            ),
        ),
        "office_tools": (
            FamilyInfo("embodiedgen", "dataset/desktop_supplies/office_tools", "Office tools from EmbodiedGen"),
        ),
        "remote_control": (
            FamilyInfo("embodiedgen", "dataset/desktop_supplies/remote_control", "Remote controls from EmbodiedGen"),
        ),
    }

    _HF_API: HfApi | None = None

    @classmethod
    def register_repository(
        cls,
        name: str,
        repo_id: str,
        repo_type: str = "dataset",
        local_root: str | Path | None = None,
        remote_root: str | Path = "",
    ) -> None:
        """Register a new asset repository.

        Args:
            name: Short name for the repository
            repo_id: HuggingFace repository ID
            repo_type: Repository type
            local_root: Local directory for assets
            remote_root: Root path within the repository
        """
        cls.REPOSITORIES[name] = AssetRepository(
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
        repo: str | None = None,
        max_assets: int | None = None,
        warn_missing: bool = True,
        use_remote_manifest: bool = True,
    ) -> list[str]:
        """Get USD/URDF assets from a family.

        Returns paths from HuggingFace manifest (if available) or local scan (fallback).
        Actual download happens on-demand when assets are used.

        Args:
            name: Family name (e.g., 'table', 'chair', 'kujiale')
            repo: Optional repository name to filter by (if None, uses all repositories)
            max_assets: Optional limit on number of assets
            warn_missing: If True, warn about missing assets
            use_remote_manifest: If True (default), query HuggingFace for complete list.
                                If False, only scan local directory.

        Returns:
            List of USD/URDF file paths (URDF files will be converted to USD at runtime)
        """
        family_key = name.lower()
        infos = cls.FAMILY_REGISTRY.get(family_key)
        if not infos:
            known = ", ".join(sorted(cls.FAMILY_REGISTRY.keys()))
            raise KeyError(f"Unknown USD family '{name}'. Available families: {known}.")

        collected: list[str] = []
        for info in infos:
            # Filter by repository if specified
            if repo is not None and info.repo != repo:
                continue

            collected.extend(
                cls._collect_usds_from_path(info.repo, info.path, max_assets, warn_missing, use_remote_manifest)
            )

        # Deduplicate and sort
        return sorted(dict.fromkeys(collected))

    @classmethod
    def _collect_usds_from_path(
        cls,
        repo_name: str,
        rel_path: str,
        max_assets: int | None,
        warn_missing: bool,
        use_remote_manifest: bool = True,
    ) -> list[str]:
        """Collect USD/URDF files from a repository path.

        Supports both USD formats (.usd, .usda, .usdz) and URDF format (.urdf).
        """
        if repo_name not in cls.REPOSITORIES:
            raise ValueError(f"Unknown repository '{repo_name}'. Available: {list(cls.REPOSITORIES.keys())}")

        repo = cls.REPOSITORIES[repo_name]
        search_path = repo.local_root / rel_path

        # Check if user wants remote manifest
        if use_remote_manifest:
            # Try remote manifest first (gets complete list from HuggingFace)
            remote_paths = cls._collect_remote_usd_paths(repo_name, rel_path)

            if remote_paths:
                # Use remote manifest (complete list, files may not exist locally yet)
                usd_paths = remote_paths
            elif search_path.exists():
                # Fallback to local scan (offline mode)
                usd_paths = []
                for ext in ["*.usd", "*.usda", "*.usdz", "*.urdf"]:
                    usd_paths.extend(sorted([str(p) for p in search_path.rglob(ext)]))
            else:
                # No remote and no local
                usd_paths = []
                if warn_missing:
                    warnings.warn(
                        f"USD/URDF assets not found at {search_path}. Download from https://huggingface.co/{repo.repo_id}",
                        stacklevel=2,
                    )
        else:
            # Local-only mode (user disabled remote)
            if search_path.exists():
                usd_paths = []
                for ext in ["*.usd", "*.usda", "*.usdz", "*.urdf"]:
                    usd_paths.extend(sorted([str(p) for p in search_path.rglob(ext)]))
            else:
                usd_paths = []
                if warn_missing:
                    warnings.warn(
                        f"USD/URDF assets not found locally at {search_path}. Set use_remote_manifest=True to query HuggingFace.",
                        stacklevel=2,
                    )

        usd_paths = sorted(dict.fromkeys(usd_paths))
        if max_assets is not None and len(usd_paths) > max_assets:
            usd_paths = usd_paths[:max_assets]

        return usd_paths

    @classmethod
    def _collect_remote_usd_paths(cls, repo_name: str, rel_path: str) -> list[str]:
        """Get USD/URDF paths from remote repository manifest.

        Supports both USD formats (.usd, .usda, .usdz) and URDF format (.urdf).
        URDF files will be automatically converted to USD at runtime by IsaacSim.

        Args:
            repo_name: Repository name
            rel_path: Relative path within the repository

        Returns:
            List of local paths to USD/URDF files
        """
        manifest = cls._remote_manifest(repo_name)
        if not manifest:
            return []

        repo = cls.REPOSITORIES[repo_name]
        remote_prefix = (repo.remote_root / rel_path).as_posix()
        normalized_prefix = remote_prefix.rstrip("/")

        if not normalized_prefix:
            prefix = ""
        else:
            prefix = normalized_prefix + "/"

        candidates = []
        # Support both USD and URDF formats
        supported_extensions = [".usd", ".usda", ".usdz", ".urdf"]
        for path in manifest:
            if path.startswith(prefix) and any(path.endswith(ext) for ext in supported_extensions):
                candidates.append(path)

        collected: list[str] = []
        for remote_path in sorted(candidates):
            try:
                if repo.remote_root:
                    relative = Path(remote_path).relative_to(repo.remote_root)
                else:
                    relative = Path(remote_path)
                local_path = repo.local_root / relative
                collected.append(str(local_path))
            except ValueError:
                continue

        return collected

    @classmethod
    @lru_cache(maxsize=8)
    def _remote_manifest(cls, repo_name: str) -> tuple[str, ...]:
        """Fetch file list from HuggingFace repository (cached)."""
        if repo_name not in cls.REPOSITORIES:
            return ()

        repo = cls.REPOSITORIES[repo_name]
        api = cls._get_hf_api()
        if api is None:
            return ()

        try:
            files = api.list_repo_files(repo_id=repo.repo_id, repo_type=repo.repo_type)
            return tuple(sorted(files))
        except Exception as exc:
            warnings.warn(f"Failed to query HuggingFace repo '{repo.repo_id}': {exc}", stacklevel=2)
            return ()

    @classmethod
    def _get_hf_api(cls) -> HfApi | None:
        """Get or create HuggingFace API instance."""
        if HfApi is None:
            return None
        if cls._HF_API is None:
            cls._HF_API = HfApi()
        return cls._HF_API

    @classmethod
    def families(cls) -> dict[str, tuple[FamilyInfo, ...]]:
        """Expose the family registry for inspection.

        Returns:
            Dictionary mapping family names to their FamilyInfo tuples
        """
        return {name: tuple(infos) for name, infos in cls.FAMILY_REGISTRY.items()}

    @classmethod
    def available_families(cls) -> list[str]:
        """Get list of all available family names.

        Returns:
            Sorted list of family names
        """
        return sorted(cls.FAMILY_REGISTRY.keys())


# =============================================================================
# Scene USD Collections
# =============================================================================


class SceneUSDCollections:
    """Curated USD asset collections organized by scene element type.

    Mirrors SceneMaterialCollections structure for USD assets. Provides both
    general-purpose methods (table_assets, scene_assets, object_assets) and
    convenience methods for specific curated collections (table785, kujiale_scenes,
    desktop_supplies).

    Example:
        >>> # Get table assets (general)
        >>> tables = SceneUSDCollections.table_assets(max_assets=10)
        >>>
        >>> # Get object assets (general)
        >>> objects = SceneUSDCollections.object_assets(families=("office_stationery", "decorations"), max_assets=20)
        >>>
        >>> # Get Table785 curated set (specific - 5 tables)
        >>> table785 = SceneUSDCollections.table785(indices=[0, 1, 2])
        >>>
        >>> # Get Kujiale scenes (specific - 12 scenes)
        >>> scenes = SceneUSDCollections.kujiale_scenes()
        >>>
        >>> # Get desktop supplies objects (specific - 10 office stationery)
        >>> desktop_objects = SceneUSDCollections.desktop_supplies(indices=[0, 1, 2])
    """

    TABLE_FAMILIES = ("table",)
    SCENE_FAMILIES = ("kujiale",)
    OBJECT_FAMILIES = ("decorations", "office_stationery", "office_tools", "remote_control")

    @staticmethod
    def table_assets(
        *,
        families: tuple[str, ...] | None = None,
        max_assets: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return table/workspace USD assets from the USD family registry.

        Args:
            families: USD families to source from (default: table)
            max_assets: Optional limit on number of assets
            warn_missing: If True, warn about missing assets

        Returns:
            List of USD file paths
        """
        return _collect_family_assets(
            families or SceneUSDCollections.TABLE_FAMILIES,
            max_assets=max_assets,
            warn_missing=warn_missing,
        )

    @staticmethod
    def scene_assets(
        *,
        families: tuple[str, ...] | None = None,
        max_assets: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return full scene USD assets from the USD family registry.

        Args:
            families: USD families to source from (default: kujiale)
            max_assets: Optional limit on number of assets
            warn_missing: If True, warn about missing assets

        Returns:
            List of USD file paths
        """
        return _collect_family_assets(
            families or SceneUSDCollections.SCENE_FAMILIES,
            max_assets=max_assets,
            warn_missing=warn_missing,
        )

    @staticmethod
    def object_assets(
        *,
        families: tuple[str, ...] | None = None,
        max_assets: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return desktop object USD assets from the USD family registry.

        Includes decorations, office_stationery, office supplies, tools, and remote controls
        from the EmbodiedGen desktop_supplies dataset.

        Args:
            families: USD families to source from (default: all object families)
            max_assets: Optional limit on number of assets
            warn_missing: If True, warn about missing assets

        Returns:
            List of USD file paths
        """
        return _collect_family_assets(
            families or SceneUSDCollections.OBJECT_FAMILIES,
            max_assets=max_assets,
            warn_missing=warn_missing,
        )

    # Convenience methods for specific curated collections
    @staticmethod
    def table785(
        *,
        indices: list[int] | None = None,
        return_configs: bool = False,
    ) -> list[str] | tuple[list[str], dict[str, dict]]:
        """Get Table785 curated set (5 specific table models from EmbodiedGen).

        This is a convenience method that returns a hardcoded list of 5 tables.
        For general table access, use table_assets() instead.

        Note: Default per-table configurations (position/scale/rotation) are available
        via `get_table_configs()` or by setting `return_configs=True`.

        Args:
            indices: Optional list of indices to select specific tables (0-4)
            return_configs: If True, returns (paths, configs) tuple where configs
                          contains default per-table calibrations

        Returns:
            List of USD file paths, or (paths, configs) tuple if return_configs=True

        Example:
            >>> # Get paths with default configs (convenient!)
            >>> paths, configs = SceneUSDCollections.table785(return_configs=True)
            >>> USDAssetPoolCfg(usd_paths=paths, per_path_overrides=configs)
        """
        TABLE785_UUIDS = (
            "126f60baf12759ea957fb6c38ba7458d",
            "1522dad65f0859758dad5636ba348bf8",
            "18848428c54456aa82070f2fd33f7bb4",
            "848396479c0b5da3bc05d0ef74d4dcfb",
            "b4b40966ebda5393bd4d7fc634062519",
        )
        paths = _collect_table785_assets(
            uuids=TABLE785_UUIDS,
            indices=indices,
        )

        if return_configs:
            # Get all table configs and filter to match selected paths
            all_configs = get_table_configs()
            configs = _filter_configs_by_paths(all_configs, paths)
            return (paths, configs)
        return paths

    @staticmethod
    def kujiale_scenes(
        *,
        indices: list[int] | None = None,
        return_configs: bool = False,
    ) -> list[str] | tuple[list[str], dict[str, dict]]:
        """Get Kujiale curated set (12 specific interior scenes from RoboVerse).

        This is a convenience method that returns a hardcoded list of 12 scenes.
        For general scene access, use scene_assets() instead.

        Note: Default per-scene configurations (position/scale/rotation) are available
        via `get_kujiale_scenes_config()` or by setting `return_configs=True`.

        Args:
            indices: Optional list of indices to select specific scenes (0-11)
            return_configs: If True, returns (paths, configs) tuple where configs
                          contains default per-scene calibrations

        Returns:
            List of USDA file paths, or (paths, configs) tuple if return_configs=True

        Example:
            >>> # Get paths only
            >>> paths = SceneUSDCollections.kujiale_scenes()
            >>>
            >>> # Get paths with default configs (convenient!)
            >>> paths, configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
            >>> USDAssetPoolCfg(usd_paths=paths, per_path_overrides=configs)
        """
        KUJIALE_INDICES = (3, 4, 8, 9, 20, 21, 22, 24, 25, 31, 32, 33)
        paths = _collect_kujiale_scenes(
            scene_indices=KUJIALE_INDICES,
            indices=indices,
        )

        if return_configs:
            # Get all scene configs and filter to match selected paths
            all_configs = get_kujiale_scenes_config()
            configs = _filter_configs_by_paths(all_configs, paths)
            return (paths, configs)
        return paths

    @staticmethod
    def desktop_supplies(
        *,
        indices: list[int] | None = None,
        return_configs: bool = False,
    ) -> list[str] | tuple[list[str], dict[str, dict]]:
        """Get desktop supplies curated set (10 specific office stationery objects from EmbodiedGen).

        This is a convenience method that returns a hardcoded list of 10 office stationery objects
        from the desktop_supplies/office_stationery category. This demonstrates the specific pattern
        similar to table785() and kujiale_scenes().
        For general object access with flexible category selection, use object_assets() instead.

        Note: Default per-object configurations (scale) are available
        via `get_desktop_object_configs()` or by setting `return_configs=True`.

        Args:
            indices: Optional list of indices to select specific objects (0-9)
            return_configs: If True, returns (paths, configs) tuple where configs
                          contains default per-object calibrations

        Returns:
            List of USD file paths, or (paths, configs) tuple if return_configs=True

        Example:
            >>> # Get paths with default configs (convenient!)
            >>> paths, configs = SceneUSDCollections.desktop_supplies(return_configs=True)
            >>> USDAssetPoolCfg(usd_paths=paths, per_path_overrides=configs)
        """
        # Curated desktop supplies: 10 office stationery from EmbodiedGen
        # Source: https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData/tree/main/dataset/desktop_supplies/office_stationery
        DESKTOP_SUPPLIES_UUIDS = {
            "office_stationery": (
                "0634f388c3845f1e929f367581352d20",
                "0773f8fc18b45b85a3a5a65c99e746e6",
                "09b4f19c0be9527883c97921b7f5d736",
                "10ab616ea78652a8a5611334723ad931",
                "1695ee4d1917544cb55ab8477ede5060",
                "1ad9e289b3f35c4e94bea6fdcc794af3",
                "1c3090016ed053e2bb444e9470aaf9cb",
                "1e7cfd9a38ca56a891842b62f92cebfa",
                "1f9eb044e10857beb7fa41f71d738e7a",
                "2faedaa2fd0d580a8c00f8d94877c446",
            ),
        }
        paths = _collect_desktop_supplies(
            curated_uuids=DESKTOP_SUPPLIES_UUIDS,
            indices=indices,
        )

        if return_configs:
            # Get all object configs and filter to match selected paths
            all_configs = get_desktop_object_configs()
            configs = _filter_configs_by_paths(all_configs, paths)
            return (paths, configs)
        return paths


def _collect_family_assets(
    families: tuple[str, ...],
    *,
    max_assets: int | None,
    warn_missing: bool,
) -> list[str]:
    """Aggregate unique USD asset paths from the given USD families."""
    paths: list[str] = []
    for family in families:
        paths.extend(
            USDCollections.family(
                family,
                max_assets=max_assets,
                warn_missing=warn_missing,
            )
        )

    unique = sorted(dict.fromkeys(paths))
    if max_assets is not None and max_assets > 0 and len(unique) > max_assets:
        unique = unique[:max_assets]
    return unique


def _collect_table785_assets(
    uuids: tuple[str, ...],
    *,
    indices: list[int] | None,
) -> list[str]:
    """Collect Table785 asset paths from EmbodiedGen repository.

    Returns USD paths if available (after conversion), otherwise URDF paths.

    Note: When auto_download=True, this function does NOT download immediately.
    Instead, it only returns the paths. Actual download happens on-demand when
    SceneRandomizer tries to load the asset, mimicking Material's behavior.

    HuggingFace only contains URDF files. To get USD:
    1. Run: python roboverse_pack/asset/download_table785_assets.py
    2. Or let IsaacSim convert URDF to USD at runtime
    """
    repo = USDCollections.REPOSITORIES["embodiedgen"]
    selected_uuids = [uuids[i] for i in indices] if indices is not None else list(uuids)

    paths = []
    for uuid in selected_uuids:
        # Prefer USD if it exists (after conversion)
        usd_rel_path = f"dataset/basic_furniture/table/{uuid}/usd/{uuid}.usd"
        usd_local_path = repo.local_root / usd_rel_path

        # Fallback to URDF (always available on HuggingFace)
        urdf_rel_path = f"dataset/basic_furniture/table/{uuid}/{uuid}.urdf"
        urdf_local_path = repo.local_root / urdf_rel_path

        # Choose which path to return
        if usd_local_path.exists():
            paths.append(str(usd_local_path))
        else:
            # Return URDF path (will be downloaded on-demand if auto_download=True)
            paths.append(str(urdf_local_path))

    return paths


def _collect_kujiale_scenes(
    scene_indices: tuple[int, ...],
    *,
    indices: list[int] | None,
) -> list[str]:
    """Collect Kujiale scene USD asset paths from RoboVerse repository.

    Files are downloaded to mirror the remote structure:
    Remote: scenes/kujiale/003.usda
    Local:  roboverse_data/scenes/kujiale/003.usda

    Note: When auto_download=True, this function does NOT download immediately.
    Instead, it only returns the paths. Actual download happens on-demand when
    SceneRandomizer tries to load the asset, mimicking Material's behavior.
    """
    repo = USDCollections.REPOSITORIES["roboverse"]
    selected_indices = [scene_indices[i] for i in indices] if indices is not None else list(scene_indices)

    paths = []
    for scene_idx in selected_indices:
        # Mirror remote structure completely
        remote_path = f"scenes/kujiale/{scene_idx:03d}.usda"
        local_path = repo.local_root / remote_path
        paths.append(str(local_path))

        # Note: Do NOT download here! Download happens on-demand in SceneRandomizer

    return paths


def _collect_desktop_supplies(
    curated_uuids: dict[str, tuple[str, ...]],
    *,
    indices: list[int] | None,
) -> list[str]:
    """Collect desktop supplies asset paths from EmbodiedGen repository.

    Returns a curated set of objects from desktop_supplies dataset.
    Currently includes 10 office stationery items. Can be extended with more categories.

    Returns USD paths if available (after conversion), otherwise URDF paths.

    Note: When auto_download=True, this function does NOT download immediately.
    Instead, it only returns the paths. Actual download happens on-demand when
    SceneRandomizer tries to load the asset, mimicking Material's behavior.

    HuggingFace only contains URDF files. To get USD:
    1. Run URDF → USD conversion locally
    2. Or let IsaacSim convert URDF to USD at runtime

    Args:
        curated_uuids: Dictionary mapping category names to UUID tuples
        indices: Optional list of indices to select specific objects
    Returns:
        List of asset file paths (USD or URDF)
    """
    repo = USDCollections.REPOSITORIES["embodiedgen"]

    # Flatten all UUIDs into a single ordered list (by category order)
    all_uuids: list[tuple[str, str]] = []  # (category, uuid)
    for category, uuids in curated_uuids.items():
        for uuid in uuids:
            all_uuids.append((category, uuid))

    # Select based on indices
    selected = [all_uuids[i] for i in indices] if indices is not None else all_uuids

    paths = []
    for category, uuid in selected:
        # Prefer USD if it exists (after conversion)
        usd_rel_path = f"dataset/desktop_supplies/{category}/{uuid}/usd/{uuid}.usd"
        usd_local_path = repo.local_root / usd_rel_path

        # Fallback to URDF (always available on HuggingFace)
        urdf_rel_path = f"dataset/desktop_supplies/{category}/{uuid}/{uuid}.urdf"
        urdf_local_path = repo.local_root / urdf_rel_path

        # Choose which path to return
        if usd_local_path.exists():
            paths.append(str(usd_local_path))
        else:
            # Return URDF path (will be downloaded on-demand if auto_download=True)
            paths.append(str(urdf_local_path))

    return paths


# =============================================================================
# Default Configurations for Curated Collections
# =============================================================================


def get_kujiale_scenes_config() -> dict[str, dict]:
    """Get all known Kujiale scene configurations (not limited to curated set).

    This is a comprehensive configuration database that includes calibrations for
    all manually tested Kujiale scenes. Individual methods (like `kujiale_scenes()`)
    will automatically filter this to match their selected scenes.

    Each scene has been manually calibrated for optimal viewing and positioning.
    Returns a dictionary mapping scene basename to its config overrides.

    Returns:
        Dictionary mapping scene file names (e.g., "003.usda") to their config overrides
        (position, rotation, scale, etc.)

    Note:
        - This can include more scenes than just the table785 curated set
        - Configurations are automatically matched by filename or UUID in path
        - Missing configurations will use default values (no error)

    Example:
        ```python
        # Automatically filtered by kujiale_scenes()
        paths, configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
        ```
    """
    return {
        # Configurations based on manually tuned values from roboverse_pack/scenes/kujiale_scene_*_cfg.py
        # These positions have been carefully calibrated for optimal viewing and robot-table alignment
        # rotation: (w, x, y, z) format - all use identity quaternion (1, 0, 0, 0)
        "003.usda": {
            "position": (2.0, 1.8, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "004.usda": {
            "position": (-3.0, 1.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "008.usda": {
            "position": (-7.2, -1.5, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "009.usda": {
            "position": (3.2, -2.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "020.usda": {
            "position": (2.0, -1.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "021.usda": {
            "position": (-5.8, 1.8, 0.0),  # Not yet manually tuned
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "022.usda": {
            "position": (-1.0, 1.1, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "024.usda": {
            "position": (1.5, 2.6, 0.0),  # Not yet manually tuned
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "025.usda": {
            "position": (2.4, 5.7, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "031.usda": {
            "position": (4.0, -9.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "032.usda": {
            "position": (0.7, -1.1, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        "033.usda": {
            "position": (0.4, -7.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        },
        # Additional scenes (can be added as they are tested)
        # Template for new scenes:
        # "XXX.usda": {
        #     "position": (x, y, 0.0),
        #     "rotation": (1.0, 0.0, 0.0, 0.0),
        #     "scale": (1.0, 1.0, 1.0),
        # },
    }


def get_table_configs() -> dict[str, dict]:
    """Get all known table configurations (not limited to Table785).

    This is a comprehensive configuration database that includes calibrations for
    all manually tested tables from EmbodiedGen. Individual methods (like `table785()`)
    will automatically filter this to match their selected UUIDs.

    Each table has been manually calibrated for optimal positioning.
    Returns a dictionary mapping table UUID to its config overrides.

    Returns:
        Dictionary mapping table UUIDs to their config overrides

    Note:
        - This can include more tables than just the Table785 curated set
        - Configurations are automatically matched by UUID found in path
        - Missing configurations will use default values (no error)

    Example:
        ```python
        # Automatically filtered by table785()
        paths, configs = SceneUSDCollections.table785(return_configs=True)
        ```
    """
    return {
        # Table785 curated set - configurations based on roboverse_pack/asset/table785_config.py
        # These scales have been manually calibrated for optimal proportions
        # Position is kept at (0, 0, 0) as it will be dynamically calculated by get_table_bounds()
        # rotation: (w, x, y, z) format - all use identity quaternion (1, 0, 0, 0)
        "126f60baf12759ea957fb6c38ba7458d": {  # Table 1
            "position": (0.0, 0.0, 0.37),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.2, 1.5, 1.0),
        },
        "1522dad65f0859758dad5636ba348bf8": {  # Table 2
            "position": (0.3, 0.0, 0.37),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.2, 1.4, 1.0),
        },
        "18848428c54456aa82070f2fd33f7bb4": {  # Table 3
            "position": (0.3, 0.0, 0.37),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.2, 1.6, 1.0),
        },
        "848396479c0b5da3bc05d0ef74d4dcfb": {  # Table 4
            "position": (0.3, 0.0, 0.37),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (2.0, 1.6, 1.0),
        },
        "b4b40966ebda5393bd4d7fc634062519": {  # Table 5
            "position": (0.3, 0.0, 0.37),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (1.3, 1.3, 1.0),
        },
        # Additional tables from EmbodiedGen (can be added as they are tested)
        # Template for new tables:
        # "uuid_here": {
        #     "position": (0.0, 0.0, 0.0),
        #     "rotation": (1.0, 0.0, 0.0, 0.0),
        #     "scale": (sx, sy, sz),
        # },
    }


def get_desktop_object_configs() -> dict[str, dict]:
    """Get all known desktop object configurations (not limited to curated set).

    This is a comprehensive configuration database that includes calibrations for
    all manually tested desktop objects from EmbodiedGen. Individual methods
    (like `desktop_supplies()`) will automatically filter this to match their selected UUIDs.

    Each object has been manually calibrated for optimal positioning on table surface.
    Returns a dictionary mapping object UUID to its config overrides.

    Returns:
        Dictionary mapping object UUIDs to their config overrides

    Note:
        - This can include objects from all categories (office_stationery, decorations, etc.)
        - Configurations are automatically matched by UUID found in path
        - Missing configurations will use default values (no error)

    Example:
        ```python
        # Automatically filtered by desktop_supplies()
        paths, configs = SceneUSDCollections.desktop_supplies(return_configs=True)
        ```
    """
    return {
        # Office stationery (curated set - 10 objects)
        # Positioned at table surface (z=0.75)
        # Static colliders: have collision but no dynamic physics (cannot fall)
        # This allows randomization between demos without PhysX errors
        # Layout: distributed around the table edges, away from task area
        # Left front area
        "0634f388c3845f1e929f367581352d20": {
            "position": (-0.5, 0.4, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Right front area
        "0773f8fc18b45b85a3a5a65c99e746e6": {
            "position": (0.5, 0.4, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Right side
        "09b4f19c0be9527883c97921b7f5d736": {
            "position": (0.6, 0.0, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Left side
        "10ab616ea78652a8a5611334723ad931": {
            "position": (-0.6, 0.0, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Center front
        "1695ee4d1917544cb55ab8477ede5060": {
            "position": (0.0, 0.5, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Far left front
        "1ad9e289b3f35c4e94bea6fdcc794af3": {
            "position": (-0.7, 0.3, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Far right front
        "1c3090016ed053e2bb444e9470aaf9cb": {
            "position": (0.7, 0.3, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Right center
        "1e7cfd9a38ca56a891842b62f92cebfa": {
            "position": (0.5, 0.1, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Left front near
        "1f9eb044e10857beb7fa41f71d738e7a": {
            "position": (-0.4, 0.5, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
        # Right front near
        "2faedaa2fd0d580a8c00f8d94877c446": {
            "position": (0.4, 0.5, 0.75),
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "scale": (0.8, 0.8, 0.8),
        },
    }


def _filter_configs_by_paths(all_configs: dict[str, dict], paths: list[str]) -> dict[str, dict]:
    """Filter configuration dictionary to only include entries matching the given paths.

    This helper function enables decoupling: the configuration database can be large
    (containing all known assets), while individual methods only return configs for
    their selected paths.

    Args:
        all_configs: Full configuration dictionary (e.g., from get_table_configs())
        paths: List of asset paths to filter for

    Returns:
        Filtered configuration dictionary containing only matching entries

    Note:
        - Matching is done by checking if config key appears in path
        - Supports both basename matching ("003.usda") and UUID matching
        - Missing paths are simply omitted (no error)
    """
    from pathlib import Path

    filtered = {}
    for path in paths:
        path_str = str(path)
        path_basename = Path(path).name

        # Try to find matching config
        for config_key, config_value in all_configs.items():
            # Match by basename or by key appearing anywhere in path
            if config_key == path_basename or config_key in path_str:
                # Use path as key (not config_key) for per_path_overrides
                filtered[path_basename] = config_value
                break

    return filtered


# =============================================================================
# Scene Material Collections
# =============================================================================


class SceneMaterialCollections:
    """Curated material collections organized by surface type."""

    TABLE_FAMILIES = ("wood", "stone", "plastic", "ceramic", "metal")
    FLOOR_FAMILIES = ("carpet", "wood", "stone", "concrete", "plastic")
    WALL_FAMILIES = ("architecture", "wall_board", "masonry", "paint", "composite")
    CEILING_FAMILIES = ("architecture", "wall_board", "wood")
    OBJECT_FAMILIES = ("wood", "metal", "plastic", "ceramic", "paper", "fabric", "stone")

    @staticmethod
    def table_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return table/desktop materials sourced from the MDL family registry.

        Args:
            families: Material families to source from (default: wood, stone, plastic, ceramic, metal)
            max_materials: Optional limit on number of materials
            warn_missing: If True, warn about missing materials

        Returns:
            List of MDL material file paths
        """
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
        """Return floor materials sourced from the MDL family registry.

        Args:
            families: Material families to source from (default: carpet, wood, stone, concrete, plastic)
            max_materials: Optional limit on number of materials
            warn_missing: If True, warn about missing materials

        Returns:
            List of MDL material file paths
        """
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
        """Return wall materials sourced from the MDL family registry.

        Args:
            families: Material families to source from (default: architecture, wall_board, masonry, paint, composite)
            max_materials: Optional limit on number of materials
            warn_missing: If True, warn about missing materials

        Returns:
            List of MDL material file paths
        """
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
        """Return ceiling materials sourced from the MDL family registry.

        Args:
            families: Material families to source from (default: architecture, wall_board, wood)
            max_materials: Optional limit on number of materials
            warn_missing: If True, warn about missing materials

        Returns:
            List of MDL material file paths
        """
        return _collect_family_materials(
            families or SceneMaterialCollections.CEILING_FAMILIES,
            max_materials=max_materials,
            warn_missing=warn_missing,
        )

    @staticmethod
    def object_materials(
        *,
        families: tuple[str, ...] | None = None,
        max_materials: int | None = None,
        warn_missing: bool = False,
    ) -> list[str]:
        """Return object materials for distractors and props sourced from the MDL family registry.

        Args:
            families: Material families to source from (default: wood, metal, plastic, ceramic, paper, fabric, stone)
            max_materials: Optional limit on number of materials
            warn_missing: If True, warn about missing materials

        Returns:
            List of MDL material file paths
        """
        return _collect_family_materials(
            families or SceneMaterialCollections.OBJECT_FAMILIES,
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
    ) -> SceneRandomCfg:
        """Create an empty room with floor, walls, and ceiling.

        Args:
            room_size: Size of the room (square)
            wall_height: Height of walls
            wall_thickness: Thickness of walls
        Returns:
            Scene randomization configuration
        """
        half_room = room_size / 2.0
        half_thickness = wall_thickness / 2.0

        return SceneRandomCfg(
            environment_layer=EnvironmentLayerCfg(
                elements=[
                    # Floor
                    ManualGeometryCfg(
                        name="floor",
                        geometry_type="cube",
                        size=(room_size, room_size, wall_thickness),
                        position=(0.0, 0.0, 0.005),
                        default_material="roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl",
                        add_collision=True,
                    ),
                    # Front wall (positive Y)
                    ManualGeometryCfg(
                        name="wall_front",
                        geometry_type="cube",
                        size=(room_size + 2 * wall_thickness, wall_thickness, wall_height),
                        position=(0.0, half_room + half_thickness, wall_height / 2),
                        default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                        add_collision=True,
                    ),
                    # Back wall (negative Y)
                    ManualGeometryCfg(
                        name="wall_back",
                        geometry_type="cube",
                        size=(room_size + 2 * wall_thickness, wall_thickness, wall_height),
                        position=(0.0, -half_room - half_thickness, wall_height / 2),
                        default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                        add_collision=True,
                    ),
                    # Left wall (negative X)
                    ManualGeometryCfg(
                        name="wall_left",
                        geometry_type="cube",
                        size=(wall_thickness, room_size, wall_height),
                        position=(-half_room - half_thickness, 0.0, wall_height / 2),
                        default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                        add_collision=True,
                    ),
                    # Right wall (positive X)
                    ManualGeometryCfg(
                        name="wall_right",
                        geometry_type="cube",
                        size=(wall_thickness, room_size, wall_height),
                        position=(half_room + half_thickness, 0.0, wall_height / 2),
                        default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                        add_collision=True,
                    ),
                    # Ceiling
                    ManualGeometryCfg(
                        name="ceiling",
                        geometry_type="cube",
                        size=(room_size, room_size, wall_thickness),
                        position=(0.0, 0.0, wall_height + wall_thickness / 2),
                        default_material="roboverse_data/materials/arnold/Architecture/Roof_Tiles.mdl",
                        add_collision=True,
                    ),
                ],
            ),
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
        # Get empty room configuration
        cfg = ScenePresets.empty_room(
            room_size=room_size,
            wall_height=wall_height,
        )

        # Add workspace layer with table
        cfg.workspace_layer = WorkspaceLayerCfg(
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=table_size,
                    position=(0.0, 0.0, table_height - table_size[2] / 2),
                ),
            ],
        )

        return cfg

    @staticmethod
    def floor_only(
        floor_size: float = 10.0,
        floor_thickness: float = 0.1,
    ) -> SceneRandomCfg:
        """Create only a floor (minimal scene).

        Args:
            floor_size: Size of the floor (square)
            floor_thickness: Thickness of floor
        Returns:
            Scene randomization configuration
        """
        return SceneRandomCfg(
            environment_layer=EnvironmentLayerCfg(
                elements=[
                    ManualGeometryCfg(
                        name="floor",
                        geometry_type="cube",
                        size=(floor_size, floor_size, floor_thickness),
                        position=(0.0, 0.0, 0.005),
                    ),
                ],
            ),
        )

    @staticmethod
    def kujiale_with_table785(
        scene_index: int | None = None,
        table_index: int | None = None,
        randomize_scene: bool = True,
        randomize_table: bool = True,
        use_default_configs: bool = True,
    ) -> SceneRandomCfg:
        """Create Kujiale scene with Table785 tables.

        Args:
            scene_index: Fixed scene index (if randomize_scene=False)
            table_index: Fixed table index (if randomize_table=False)
            randomize_scene: If True, randomly select from all Kujiale scenes
            randomize_table: If True, randomly select from all Table785 tables
            use_default_configs: If True, apply default per-path position/scale configurations
                                 for each scene/table

        Returns:
            Scene randomization configuration with USD assets
        """
        # Get paths and configs from SceneUSDCollections
        # Configs are automatically filtered to match selected paths
        if use_default_configs:
            # Convenient: get paths and configs together (auto-filtered)
            scene_paths, scene_configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
            table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)
        else:
            # Get paths only
            scene_paths = SceneUSDCollections.kujiale_scenes()
            table_paths = SceneUSDCollections.table785()
            scene_configs = None
            table_configs = None

        # Create environment layer with Kujiale scene
        if randomize_scene:
            env_element = USDAssetPoolCfg(
                name="kujiale_scene",
                usd_paths=scene_paths,
                position=(0.0, 0.0, 0.0),
                per_path_overrides=scene_configs,
                selection_strategy="random",
            )
        else:
            env_element = USDAssetCfg(
                name="kujiale_scene",
                usd_path=scene_paths[scene_index or 0],
                position=(0.0, 0.0, 0.0),
            )

        # Create workspace layer with table
        if randomize_table:
            workspace_element = USDAssetPoolCfg(
                name="table",
                usd_paths=table_paths,
                position=(0.0, 0.0, 0.0),  # Per-path configs will override
                scale=(1.0, 1.0, 1.0),  # Per-path configs will override
                per_path_overrides=table_configs,  # Apply per-table calibrations
                selection_strategy="random",
            )
        else:
            workspace_element = USDAssetCfg(
                name="table",
                usd_path=table_paths[table_index or 0],
                position=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
            )

        return SceneRandomCfg(
            environment_layer=EnvironmentLayerCfg(
                elements=[env_element],
            ),
            workspace_layer=WorkspaceLayerCfg(
                elements=[workspace_element],
            ),
        )

    @staticmethod
    def hybrid_scene(
        room_size: float = 10.0,
        wall_height: float = 5.0,
        kujiale_scene: str | None = None,
        table_size: tuple[float, float, float] = (1.8, 1.8, 0.1),
        table_height: float = 0.7,
        num_distractor_objects: int = 0,
        *,
        use_default_scene_config: bool = True,
    ) -> SceneRandomCfg:
        """Create a hybrid scene combining USD background with manual geometry workspace.

        Args:
            room_size: Room size for manual room (if kujiale_scene is None)
            wall_height: Wall height for manual room (if kujiale_scene is None)
            kujiale_scene: Optional Kujiale scene USD path. If None, creates manual room.
            table_size: Table size
            table_height: Table height from ground
            num_distractor_objects: Number of distractor objects to add on table
            use_default_scene_config: If True, apply default config for the Kujiale scene
                                     (looks up position/scale from get_kujiale_scenes_config)

        Returns:
            Scene randomization configuration
        """
        # Environment layer: USD scene or manual room
        if kujiale_scene:
            # Get default config for this specific scene (if available and enabled)
            scene_position = (0.0, 0.0, 0.0)
            scene_scale = (1.0, 1.0, 1.0)
            scene_rotation = (1.0, 0.0, 0.0, 0.0)

            if use_default_scene_config:
                from pathlib import Path

                scene_basename = Path(kujiale_scene).name
                scene_configs = get_kujiale_scenes_config()
                if scene_basename in scene_configs:
                    config = scene_configs[scene_basename]
                    scene_position = config.get("position", scene_position)
                    scene_scale = config.get("scale", scene_scale)
                    scene_rotation = config.get("rotation", scene_rotation)

            environment_layer = EnvironmentLayerCfg(
                elements=[
                    USDAssetCfg(
                        name="kujiale_scene",
                        usd_path=kujiale_scene,
                        position=scene_position,
                        rotation=scene_rotation,
                        scale=scene_scale,
                    ),
                ],
            )
        else:
            # Use manual room
            cfg = ScenePresets.empty_room(
                room_size=room_size,
                wall_height=wall_height,
            )
            environment_layer = cfg.environment_layer

        # Workspace layer: manual table with material randomization
        workspace_layer = WorkspaceLayerCfg(
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=table_size,
                    position=(0.0, 0.0, table_height - table_size[2] / 2),
                ),
            ],
        )

        # Objects layer: distractor objects (if requested)
        objects_layer = None
        if num_distractor_objects > 0:
            objects_elements = []
            grid_size = int(num_distractor_objects**0.5) + 1
            spacing = min(table_size[0], table_size[1]) / (grid_size + 1)

            for i in range(num_distractor_objects):
                row = i // grid_size
                col = i % grid_size
                x = -table_size[0] / 2 + (col + 1) * spacing
                y = -table_size[1] / 2 + (row + 1) * spacing
                z = table_height + 0.025

                objects_elements.append(
                    ManualGeometryCfg(
                        name=f"distractor_{i}",
                        geometry_type="cube",
                        size=(0.05, 0.05, 0.05),
                        position=(x, y, z),
                    )
                )

            objects_layer = ObjectsLayerCfg(elements=objects_elements)

        return SceneRandomCfg(
            environment_layer=environment_layer,
            workspace_layer=workspace_layer,
            objects_layer=objects_layer,
        )
