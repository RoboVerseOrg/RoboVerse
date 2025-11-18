"""Object Registry - Central registry for all objects in the simulation.

The ObjectRegistry provides a unified interface to access all objects regardless of
how they were created (Handler vs SceneRandomizer). This enables Property Randomizers
to operate on both static and dynamic objects seamlessly.

Key concepts:
- Static Objects: Created by Handler, cannot be added/removed after launch
- Dynamic Objects: Created by SceneRandomizer, can be added/removed anytime
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


@dataclass
class ObjectMetadata:
    """Metadata for a registered object.

    Attributes:
        name: Unique object name
        category: Object category (robot, object, camera, light, scene_element)
        lifecycle: Lifecycle type (static: Handler-managed, dynamic: SceneRandomizer-managed)
        prim_paths: List of USD prim paths (one per environment if not shared)
        shared: Whether the object is shared across all environments
        has_physics: Whether the object has physics properties (mass, friction, etc)
        layer: Scene layer name (for scene_element only: environment, workspace, objects)
    """

    name: str
    category: Literal["robot", "object", "camera", "light", "scene_element"]
    lifecycle: Literal["static", "dynamic"]
    prim_paths: list[str]
    shared: bool = False
    has_physics: bool = False
    layer: str | None = None


class ObjectRegistry:
    """Central registry for all simulation objects.

    This singleton class maintains metadata for all objects in the simulation,
    enabling unified access regardless of creation method.

    Usage:
        # Initialize (usually done by Handler)
        registry = ObjectRegistry.get_instance(handler)

        # Register an object
        registry.register(ObjectMetadata(
            name="franka",
            category="robot",
            lifecycle="static",
            prim_paths=["/World/envs/env_0/franka", ...],
            has_physics=True,
        ))

        # Query objects
        obj_meta = registry.get("franka")
        prim_paths = registry.get_prim_paths("franka", env_ids=[0, 1])
    """

    _instance: ObjectRegistry | None = None

    @classmethod
    def get_instance(cls, handler: BaseSimHandler | None = None) -> ObjectRegistry:
        """Get or create the singleton ObjectRegistry instance.

        Args:
            handler: SimHandler instance (required for first call)

        Returns:
            ObjectRegistry singleton instance
        """
        if cls._instance is None:
            if handler is None:
                raise RuntimeError("First call to ObjectRegistry.get_instance() must provide a handler")
            cls._instance = cls(handler)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance.

        This is useful when switching between different handlers or in testing.
        """
        cls._instance = None

    def __init__(self, handler: BaseSimHandler):
        """Initialize ObjectRegistry (internal, use get_instance() instead)."""
        self.handler = handler
        self._registry: dict[str, ObjectMetadata] = {}

    def register(self, metadata: ObjectMetadata):
        """Register an object in the registry.

        Args:
            metadata: Object metadata to register
        """
        if metadata.name in self._registry:
            # Update existing entry (for dynamic objects that may be recreated)
            self._registry[metadata.name] = metadata
        else:
            self._registry[metadata.name] = metadata

    def unregister(self, name: str):
        """Unregister an object from the registry.

        Args:
            name: Object name to unregister
        """
        if name in self._registry:
            del self._registry[name]

    def get(self, name: str) -> ObjectMetadata | None:
        """Get object metadata by name.

        Args:
            name: Object name

        Returns:
            ObjectMetadata if found, None otherwise
        """
        return self._registry.get(name)

    def get_prim_paths(self, name: str, env_ids: list[int] | None = None) -> list[str]:
        """Get USD prim paths for an object.

        Args:
            name: Object name
            env_ids: Environment IDs to get paths for (None = all environments)
                     Can be list or tensor

        Returns:
            List of USD prim paths

        Raises:
            ValueError: If object not found
        """
        import torch

        obj = self.get(name)
        if not obj:
            raise ValueError(f"Object '{name}' not found in registry. Available objects: {list(self._registry.keys())}")

        if obj.shared:
            # Shared object: only one prim path
            return [obj.prim_paths[0]]
        else:
            # Per-env object: filter by env_ids
            if env_ids is None:
                return obj.prim_paths

            # Convert tensor to list if needed
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().tolist()

            return [obj.prim_paths[i] for i in env_ids if i < len(obj.prim_paths)]

    def has_physics(self, name: str) -> bool:
        """Check if an object has physics properties.

        Args:
            name: Object name

        Returns:
            True if object has physics, False otherwise
        """
        obj = self.get(name)
        return obj.has_physics if obj else False

    def list_objects(
        self,
        category: str | None = None,
        lifecycle: str | None = None,
        has_physics: bool | None = None,
    ) -> list[str]:
        """List object names with optional filtering.

        Args:
            category: Filter by category (robot, object, camera, light, scene_element)
            lifecycle: Filter by lifecycle (static, dynamic)
            has_physics: Filter by physics capability

        Returns:
            List of object names matching the filters
        """
        objects = list(self._registry.values())

        if category:
            objects = [o for o in objects if o.category == category]
        if lifecycle:
            objects = [o for o in objects if o.lifecycle == lifecycle]
        if has_physics is not None:
            objects = [o for o in objects if o.has_physics == has_physics]

        return [o.name for o in objects]

    def __repr__(self) -> str:
        return (
            f"ObjectRegistry("
            f"objects={len(self._registry)}, "
            f"static={len(self.list_objects(lifecycle='static'))}, "
            f"dynamic={len(self.list_objects(lifecycle='dynamic'))})"
        )
