"""Base class for all randomizer types.

This module provides the foundational BaseRandomizerType class that all randomizers
inherit from. It handles:
- Random seed management
- Handler binding with automatic Hybrid support
- Visual dirty flag for tracking updates
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from .core.isaacsim_adapter import IsaacSimAdapter
from .core.object_registry import ObjectRegistry

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class BaseRandomizerType:
    """Base class for all randomizer types.

    This class provides core functionality for all randomizers:
    - Reproducible random number generation
    - Handler binding with automatic Hybrid support
    - Visual dirty flag for change tracking

    Subclasses should:
    1. Set REQUIRES_HANDLER class attribute ("render", "physics", or "any")
    2. Implement __call__() method for randomization logic
    3. Call _mark_visual_dirty() when making visual changes

    Hybrid Simulation Support:
        When bound to a HybridSimHandler, randomizers automatically dispatch to
        the correct sub-handler based on REQUIRES_HANDLER:
        - "render": Uses render_handler (for visual randomizers)
        - "physics": Uses physics_handler (for physics randomizers)
        - "any": Uses render_handler (default)
    """

    # Class attribute: Specifies which handler this randomizer requires
    # Subclasses should override this
    REQUIRES_HANDLER: Literal["render", "physics", "any"] = "any"

    def __init__(self, *, seed: int | None = None, **kwargs):
        """Initialize base randomizer.

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional options stored in randomizer_options
        """
        self.handler: BaseSimHandler | None = None  # User-provided handler
        self._actual_handler: BaseSimHandler | None = None  # Actual handler (may be sub-handler for Hybrid)
        self.randomizer_options = kwargs
        self._seed: int | None = None
        self._rng: random.Random | None = None
        self._visual_dirty = False
        if seed is not None:
            self.set_seed(seed)

    @property
    def seed(self) -> int | None:
        """Return the current seed."""
        return self._seed

    @property
    def rng(self) -> random.Random:
        """Access internal RNG, ensuring it exists."""
        if self._rng is None:
            self.set_seed(self._seed)
        return self._rng

    def set_seed(self, seed: int | None) -> None:
        """Set or update the random seed for the randomizer.

        Args:
            seed: Seed to initialize RNG with. If None, derives from global RNG.
        """
        if seed is None:
            # Derive deterministic seed from global RNG (itself seedable)
            seed = random.getrandbits(64)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def bind_handler(self, handler: BaseSimHandler, *args: Any, **kwargs):
        """Bind handler to the randomizer with automatic Hybrid support.

        This method automatically:
        1. Detects HybridSimHandler and selects appropriate sub-handler
        2. Initializes ObjectRegistry (first bind only)
        3. Scans and registers Handler objects (first bind only)

        Args:
            handler: SimHandler instance (may be Hybrid)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.handler = handler

        # Detect and handle Hybrid
        if self._is_hybrid_handler(handler):
            self._actual_handler = self._select_hybrid_handler(handler)
            logger.debug(
                f"{self.__class__.__name__} bound to "
                f"{self._actual_handler.__class__.__name__} "
                f"(from HybridSimHandler, requires={self.REQUIRES_HANDLER})"
            )
        else:
            self._actual_handler = handler
        # Initialize IsaacSimAdapter for USD operations
        self.adapter = IsaacSimAdapter(self._actual_handler)
        # Initialize and populate ObjectRegistry (lazy, on first bind)
        self.registry = ObjectRegistry.get_instance(self._actual_handler)
        self._scan_and_register_handler_objects(self._actual_handler, self.registry)

    def _is_hybrid_handler(self, handler) -> bool:
        """Check if handler is a HybridSimHandler.

        Args:
            handler: Handler to check

        Returns:
            True if handler is Hybrid, False otherwise
        """
        return hasattr(handler, "physics_handler") and hasattr(handler, "render_handler")

    def _select_hybrid_handler(self, hybrid_handler):
        """Select appropriate sub-handler from HybridSimHandler.

        Args:
            hybrid_handler: HybridSimHandler instance

        Returns:
            Selected sub-handler based on REQUIRES_HANDLER
        """
        if self.REQUIRES_HANDLER == "render":
            return hybrid_handler.render_handler
        elif self.REQUIRES_HANDLER == "physics":
            return hybrid_handler.physics_handler
        else:
            # Default: render_handler (most randomizers are visual)
            return hybrid_handler.render_handler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Perform the randomization (implemented by subclasses)."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    # -------------------------------------------------------------------------
    # Visual dirty flag for tracking when visual updates are needed
    # -------------------------------------------------------------------------

    def _mark_visual_dirty(self) -> None:
        """Mark that visual updates have been made (for external tracking)."""
        self._visual_dirty = True

    def consume_visual_dirty(self) -> bool:
        """Check and reset the visual dirty flag.

        Returns:
            True if visual updates were made since last check
        """
        dirty = self._visual_dirty
        self._visual_dirty = False
        return dirty

    # -------------------------------------------------------------------------
    # ObjectRegistry initialization (automatic, transparent to users)
    # -------------------------------------------------------------------------

    @staticmethod
    def _scan_and_register_handler_objects(handler: BaseSimHandler, registry: ObjectRegistry) -> None:
        """Scan Handler and register all existing objects.

        Args:
            handler: BaseSimHandler
            registry: ObjectRegistry instance to populate
        """
        from metasim.randomization.core.object_registry import ObjectMetadata

        # Register robots
        if hasattr(handler, "scene") and hasattr(handler.scene, "articulations"):
            for name, obj_inst in handler.scene.articulations.items():
                prim_path = obj_inst.cfg.prim_path
                prim_paths = [prim_path.replace("env_.*", f"env_{i}") for i in range(handler.num_envs)]

                registry.register(
                    ObjectMetadata(
                        name=name,
                        category="robot",
                        lifecycle="static",
                        prim_paths=prim_paths,
                        shared=False,
                        has_physics=True,
                    )
                )

        # Register rigid objects
        if hasattr(handler, "scene") and hasattr(handler.scene, "rigid_objects"):
            for name, obj_inst in handler.scene.rigid_objects.items():
                prim_path = obj_inst.cfg.prim_path
                prim_paths = [prim_path.replace("env_.*", f"env_{i}") for i in range(handler.num_envs)]

                # Check if object has collision (indicates physics)
                has_physics = True  # Assume true for rigid objects

                registry.register(
                    ObjectMetadata(
                        name=name,
                        category="object",
                        lifecycle="static",
                        prim_paths=prim_paths,
                        shared=False,
                        has_physics=has_physics,
                    )
                )

        # Register cameras
        if hasattr(handler, "cameras"):
            for camera in handler.cameras:
                prim_path = f"/World/{camera.name}"

                registry.register(
                    ObjectMetadata(
                        name=camera.name,
                        category="camera",
                        lifecycle="static",
                        prim_paths=[prim_path],
                        shared=True,
                        has_physics=False,
                    )
                )

        # Register lights
        if hasattr(handler, "lights"):
            for light in handler.lights:
                prim_path = f"/World/{light.name}"

                registry.register(
                    ObjectMetadata(
                        name=light.name,
                        category="light",
                        lifecycle="static",
                        prim_paths=[prim_path],
                        shared=True,
                        has_physics=False,
                    )
                )
