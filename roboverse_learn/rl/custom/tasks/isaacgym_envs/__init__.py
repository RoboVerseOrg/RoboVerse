"""IsaacGymEnvs task wrappers."""

# Import all task wrappers to ensure they are registered
from .allegro_hand import AllegroHandTaskWrapper
from .ant import AntTaskWrapper
from .anymal import AnymalTaskWrapper
from .anymal_terrain import AnymalTerrainTaskWrapper
from .cartpole import CartpoleTaskWrapper

__all__ = [
    "AllegroHandTaskWrapper",
    "AntTaskWrapper",
    "AnymalTaskWrapper",
    "AnymalTerrainTaskWrapper",
    "CartpoleTaskWrapper",
]
