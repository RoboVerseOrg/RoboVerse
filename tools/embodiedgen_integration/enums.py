# ruff: noqa: UP006
# ruff: noqa: UP007

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union

from dataclasses_json import DataClassJsonMixin

__all__ = [
    "AssetType",
    "LayoutInfo",
    "Scene3DItemEnum",
]


@dataclass
class AssetType(str):
    """Asset type enumeration."""

    MJCF = "mjcf"
    USD = "usd"
    URDF = "urdf"
    MESH = "mesh"


class SimAssetMapper:
    _mapping = dict(
        ISAACSIM=AssetType.USD,
        ISAACGYM=AssetType.URDF,
        MUJOCO=AssetType.MJCF,
        GENESIS=AssetType.MJCF,
        SAPIEN=AssetType.URDF,
        PYBULLET=AssetType.URDF,
    )

    @classmethod
    def __class_getitem__(cls, key: str):
        key = key.upper()
        if key.startswith("SAPIEN"):
            key = "SAPIEN"
        return cls._mapping[key]


@dataclass
class LayoutInfo(DataClassJsonMixin):
    """Layout information for scene generation."""

    tree: Dict[str, List]
    relation: Dict[str, Union[str, List[str]]]
    objs_desc: Dict[str, str] = field(default_factory=dict)
    objs_mapping: Dict[str, str] = field(default_factory=dict)
    assets: Dict[str, str] = field(default_factory=dict)
    quality: Dict[str, str] = field(default_factory=dict)
    position: Dict[str, List[float]] = field(default_factory=dict)


class Scene3DItemEnum(str, Enum):
    """3D Scene item enumeration."""

    BACKGROUND = "background"
    CONTEXT = "context"
    ROBOT = "robot"
    MANIPULATED_OBJS = "manipulated_objs"
    DISTRACTOR_OBJS = "distractor_objs"
    OTHERS = "others"

    @classmethod
    def object_list(cls, layout_relation: dict) -> list:
        """Get the list of objects in the scene based on layout relation."""
        return (
            [
                layout_relation[cls.BACKGROUND.value],
                layout_relation[cls.CONTEXT.value],
            ]
            + layout_relation[cls.MANIPULATED_OBJS.value]
            + layout_relation[cls.DISTRACTOR_OBJS.value]
        )

    @classmethod
    def object_mapping(cls, layout_relation):
        """Get the mapping of objects in the scene based on layout relation."""
        relation_mapping = {
            # layout_relation[cls.ROBOT.value]: cls.ROBOT.value,
            layout_relation[cls.BACKGROUND.value]: cls.BACKGROUND.value,
            layout_relation[cls.CONTEXT.value]: cls.CONTEXT.value,
        }
        relation_mapping.update({
            item: cls.MANIPULATED_OBJS.value for item in layout_relation[cls.MANIPULATED_OBJS.value]
        })
        relation_mapping.update({
            item: cls.DISTRACTOR_OBJS.value for item in layout_relation[cls.DISTRACTOR_OBJS.value]
        })

        return relation_mapping
