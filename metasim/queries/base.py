"""Base class for all query types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class BaseQueryType:
    """Base class for all query types."""

    supported_handlers = []

    def __init__(self, **kwargs):
        self.handler = None
        self.query_options = kwargs

    def bind_handler(self, handler, *args: Any, **kwargs):
        """Binding handler to the query.

        By default, all queries will be binded to the handler at handler.launch() stage.
        By default, the queries will be given the handler instance as the first argument.
        For different simulation handlers, the queries may be binded at different stages.
        For different simulation handlers, the queries may be given other optional arguments.
        You can also pass locals() to bind_handler() to access local variables in the handler.
        """
        self.handler = handler
        assert self.handler.__class__.__module__ in self.supported_handlers, (
            f"Query {self} does not support handler type: {type(self.handler)}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Performing the query.

        By default, the query has no arguments.
        The query will be called in handler.get_extra() stage.
        The query should return a specific value.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
