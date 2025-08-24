"""Base class for all query types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class BaseRandomizerType:
    """Base class for all randomizer types."""

    supported_handlers = []

    def __init__(self, **kwargs):
        self.handler = None
        self.randomizer_options = kwargs

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Binding handler to the randomizer.

        """
        self.handler = handler

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Performing the query.

        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
