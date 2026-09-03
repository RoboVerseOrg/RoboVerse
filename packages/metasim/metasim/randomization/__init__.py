"""Randomization for RoboVerse.

This module provides a comprehensive domain randomization framework with:
- Core infrastructure (ObjectRegistry, IsaacSimAdapter)
- Lifecycle management (SceneRandomizer)
- Property editors (Material, Object, Light, Camera Randomizers)
- User-friendly Presets

Architecture:
- Static Objects: Handler-managed (Robot, Task Objects, Camera, Light)
- Dynamic Objects: SceneRandomizer-managed (Floor, Table, Distractors)
- Unified access via ObjectRegistry
- Hybrid simulation support (automatic handler dispatch)
"""

from metasim.randomization import *

# Randomizers
from .camera_randomizer import (
    CameraIntrinsicsRandomCfg,
    CameraLookAtRandomCfg,
    CameraOrientationRandomCfg,
    CameraPositionRandomCfg,
    CameraRandomCfg,
    CameraRandomizer,
)

# Core infrastructure
from .core import IsaacSimAdapter, ObjectMetadata, ObjectRegistry

# Unified DR Manager
from .dr_manager import DomainRandomizationManager, DRConfig
from .light_randomizer import (
    LightColorRandomCfg,
    LightIntensityRandomCfg,
    LightOrientationRandomCfg,
    LightPositionRandomCfg,
    LightRandomCfg,
    LightRandomizer,
)
from .material_randomizer import (
    MaterialRandomCfg,
    MaterialRandomizer,
    MDLMaterialCfg,
    PBRMaterialCfg,
    PhysicalMaterialCfg,
)
from .object_randomizer import ObjectRandomCfg, ObjectRandomizer, PhysicsRandomCfg, PoseRandomCfg

# Presets
from .presets import CameraPresets, LightPresets, MaterialPresets, ObjectPresets, ScenePresets
from .presets.light_presets import (
    LightColorRanges,
    LightIntensityRanges,
    LightOrientationRanges,
    LightPositionRanges,
    LightScenarios,
)
from .presets.material_presets import MaterialRepository
from .presets.scene_presets import AssetRepository, SceneUSDCollections, USDCollections
from .scene_randomizer import (
    EnvironmentLayerCfg,
    ManualGeometryCfg,
    ObjectsLayerCfg,
    SceneLayerCfg,
    SceneRandomCfg,
    SceneRandomizer,
    USDAssetCfg,
    USDAssetPoolCfg,
    WorkspaceLayerCfg,
)

__all__ = [
    "AssetRepository",
    "CameraIntrinsicsRandomCfg",
    "CameraLookAtRandomCfg",
    "CameraOrientationRandomCfg",
    "CameraPositionRandomCfg",
    "CameraPresets",
    "CameraRandomCfg",
    "CameraRandomizer",
    "DRConfig",
    "DomainRandomizationManager",
    "EnvironmentLayerCfg",
    "IsaacSimAdapter",
    "LightColorRandomCfg",
    "LightColorRanges",
    "LightIntensityRandomCfg",
    "LightIntensityRanges",
    "LightOrientationRandomCfg",
    "LightOrientationRanges",
    "LightPositionRandomCfg",
    "LightPositionRanges",
    "LightPresets",
    "LightRandomCfg",
    "LightRandomizer",
    "LightScenarios",
    "MDLMaterialCfg",
    "ManualGeometryCfg",
    "MaterialPresets",
    "MaterialRandomCfg",
    "MaterialRandomizer",
    "MaterialRepository",
    "ObjectMetadata",
    "ObjectPresets",
    "ObjectRandomCfg",
    "ObjectRandomizer",
    "ObjectRegistry",
    "ObjectsLayerCfg",
    "PBRMaterialCfg",
    "PhysicalMaterialCfg",
    "PhysicsRandomCfg",
    "PoseRandomCfg",
    "SceneLayerCfg",
    "ScenePresets",
    "SceneRandomCfg",
    "SceneRandomizer",
    "SceneUSDCollections",
    "USDAssetCfg",
    "USDAssetPoolCfg",
    "USDCollections",
    "WorkspaceLayerCfg",
]
