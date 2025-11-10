"""Type aliases for Unitree RL tasks."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .base_agent import AgentTask
    from .base_legged_robot import LeggedRobotTask

    EnvTypes = Union[AgentTask, LeggedRobotTask]
else:
    # At runtime, use a more permissive type to avoid circular imports
    from typing import Any

    EnvTypes = Any
