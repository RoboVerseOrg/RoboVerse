"""IsaacLab integration utilities.

This package provides a small compatibility layer to allow IsaacLab-style tasks
to run within MetaSim workflows (shared app lifecycle, wrappers, config conversion).
"""

from .runtime import IsaacLabRuntimeContext, ensure_isaaclab_app, release_isaaclab_app

__all__ = [
    "IsaacLabRuntimeContext",
    "ensure_isaaclab_app",
    "release_isaaclab_app",
]
