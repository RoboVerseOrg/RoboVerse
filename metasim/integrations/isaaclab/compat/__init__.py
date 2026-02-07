"""Compatibility layer for running IsaacLab manager-based tasks on MetaSim handlers."""

from .env import HandlerBackedManagerBasedRLEnv
from .registry import register_manager_based_cfg_task

__all__ = [
    "HandlerBackedManagerBasedRLEnv",
    "register_manager_based_cfg_task",
]
