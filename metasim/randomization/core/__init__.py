"""Core components for randomization system.

This module provides fundamental infrastructure for the randomization framework:
- ObjectRegistry: Central registry for all objects (static and dynamic)
- IsaacSimAdapter: Unified interface for USD operations
"""

from .isaacsim_adapter import IsaacSimAdapter
from .object_registry import ObjectMetadata, ObjectRegistry

__all__ = [
    "IsaacSimAdapter",
    "ObjectMetadata",
    "ObjectRegistry",
]
