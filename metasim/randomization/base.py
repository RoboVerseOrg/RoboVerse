"""Base class for all randomizer types."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    pass


class BaseRandomizerType:
    """Base class for all randomizer types."""

    supported_handlers = []

    def __init__(self, *, seed: int | None = None, **kwargs):
        self.handler = None
        self.randomizer_options = kwargs
        self._seed: int | None = None
        self._rng: random.Random | None = None
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
            # Derive deterministic seed from global RNG (itself seedable).
            seed = random.getrandbits(64)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Binding handler to the randomizer."""
        self.handler = handler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Performing the randomization."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    # ---------------------------------------------------------------------
    # Synchronization helpers shared by visual randomizers
    # ---------------------------------------------------------------------
    def _sync_visual_updates(self, *, wait_for_materials: bool = False, settle_passes: int = 3) -> None:
        """Ensure renderer state catches up before sensors capture new frames."""
        if wait_for_materials:
            self._wait_for_material_library()

        self._settle_renderer_frames(settle_passes=settle_passes)
        self._wait_for_async_engine()

    def _wait_for_material_library(self) -> None:
        """Block until MDL materials finish compiling (best-effort)."""
        waited = False
        try:
            import omni.kit.material.library as matlib

            wait_fn = getattr(matlib, "wait_for_pending_refreshes", None)
            if callable(wait_fn):
                wait_fn()
                waited = True
            else:
                get_instance = getattr(matlib, "get_instance", None)
                if callable(get_instance):
                    instance = get_instance()
                    for attr_name in (
                        "wait_for_pending_refreshes",
                        "wait_for_pending_compiles",
                        "wait_for_pending_compile",
                        "wait_for_pending_tasks",
                    ):
                        wait_method = getattr(instance, attr_name, None)
                        if callable(wait_method):
                            wait_method()
                            waited = True
                            break

            if not waited:
                logger.debug("Material library exposes no pending-refresh wait API; continuing")
        except ImportError:
            logger.debug("Material library not available; skipping wait")
        except Exception as err:
            logger.warning(f"Failed to wait for material refreshes: {err}")

    def _settle_renderer_frames(self, *, settle_passes: int = 3) -> None:
        """Run a few zero-dt updates + renders so sensors see the final state."""
        handler = getattr(self, "handler", None)
        if handler is None:
            return

        scene = getattr(handler, "scene", None)
        if scene is None:
            return

        sim = getattr(handler, "sim", None)
        passes = max(0, settle_passes)
        for _ in range(passes):
            try:
                scene.update(dt=0)
            except Exception as err:
                logger.debug(f"Scene update during settle failed: {err}")
                break

            if sim is not None:
                try:
                    if sim.has_gui() or sim.has_rtx_sensors():
                        sim.render()
                except Exception as err:
                    logger.debug(f"Sim render during settle failed: {err}")

            sensors = getattr(scene, "sensors", {})
            for sensor in sensors.values():
                try:
                    sensor.update(dt=0)
                except Exception as err:
                    logger.debug(f"Sensor update during settle failed: {err}")

    def _wait_for_async_engine(self) -> None:
        """Flush Kit async tasks so downstream reads observe committed data."""
        try:
            from omni.kit.async_engine import get_async_engine

            get_async_engine().wait_for_tasks()
        except ImportError:
            logger.debug("Omniverse async engine not available; skipping wait_for_tasks")
        except Exception as err:
            logger.warning(f"Failed to wait for async tasks: {err}")
