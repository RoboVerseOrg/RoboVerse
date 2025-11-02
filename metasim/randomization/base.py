"""Base class for all randomizer types."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

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
