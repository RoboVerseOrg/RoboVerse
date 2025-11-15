"""Checker operators for combining multiple checkers."""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.scenario.objects import BaseObjCfg
from metasim.utils.configclass import configclass
from metasim.utils.state import TensorState

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class AndOp:
    """Logical AND operator for checkers.

    All checkers must return True for the result to be True.
    """

    checkers: list = None
    """List of checkers to combine with AND logic."""

    def __init__(self, checkers: list):
        self.checkers = checkers

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """Reset all sub-checkers."""
        for checker in self.checkers:
            checker.reset(handler, env_ids)

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        """Check if all sub-checkers return True."""
        if not self.checkers:
            return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

        results = []
        for checker in self.checkers:
            result = checker.check(handler, states)
            results.append(result)

        # Combine all results with AND logic
        combined = torch.stack(results, dim=0)
        final_result = torch.all(combined, dim=0)

        log.debug(
            f"AndOp: {len(self.checkers)} checkers, final result: {final_result.sum()}/{final_result.numel()} True"
        )
        return final_result

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get debug viewers from all sub-checkers."""
        viewers = []
        for checker in self.checkers:
            viewers.extend(checker.get_debug_viewers())
        return viewers


@configclass
class OrOp:
    """Logical OR operator for checkers.

    At least one checker must return True for the result to be True.
    """

    checkers: list = None
    """List of checkers to combine with OR logic."""

    def __init__(self, checkers: list):
        self.checkers = checkers

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """Reset all sub-checkers."""
        for checker in self.checkers:
            checker.reset(handler, env_ids)

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        """Check if any sub-checker returns True."""
        if not self.checkers:
            return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

        results = []
        for checker in self.checkers:
            result = checker.check(handler, states)
            results.append(result)

        # Combine all results with OR logic
        combined = torch.stack(results, dim=0)
        final_result = torch.any(combined, dim=0)

        log.debug(
            f"OrOp: {len(self.checkers)} checkers, final result: {final_result.sum()}/{final_result.numel()} True"
        )
        return final_result

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get debug viewers from all sub-checkers."""
        viewers = []
        for checker in self.checkers:
            viewers.extend(checker.get_debug_viewers())
        return viewers


@configclass
class NotOp:
    """Logical NOT operator for a checker.

    Inverts the result of a single checker.
    """

    checker = None
    """The checker to invert."""

    def __init__(self, checker):
        self.checker = checker

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """Reset the sub-checker."""
        self.checker.reset(handler, env_ids)

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        """Check if the sub-checker returns False."""
        result = self.checker.check(handler, states)
        final_result = ~result

        log.debug(f"NotOp: inverted result: {final_result.sum()}/{final_result.numel()} True")
        return final_result

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get debug viewers from the sub-checker."""
        return self.checker.get_debug_viewers()


@configclass
class DrawerOpenChecker:
    """Checker to determine if a drawer is open.

    This is a specialized checker for articulated objects with drawers.
    """

    obj_name: str = None
    """Name of the cabinet object."""
    joint_name: str = None
    """Name of the drawer joint (e.g., 'bottom_drawer_joint')."""
    open_threshold: float = 0.1
    """Minimum joint position to consider the drawer as open."""

    def __init__(self, obj_name: str, joint_name: str | None = None, open_threshold: float = 0.1):
        self.obj_name = obj_name
        self.joint_name = joint_name or f"{obj_name}_bottom_drawer_joint"
        self.open_threshold = open_threshold

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """Reset the checker."""
        pass

    def check(self, handler: BaseSimHandler, states: TensorState) -> torch.BoolTensor:
        """Check if the drawer is open."""
        try:
            # Try to get joint position for the drawer
            # This is a simplified implementation - in practice you'd need to
            # query the actual joint state from the articulated object
            from .util import get_dof_pos

            joint_pos = get_dof_pos(handler, self.obj_name, self.joint_name)
            is_open = joint_pos > self.open_threshold

            log.debug(
                f"Drawer {self.joint_name} position: {joint_pos.mean():.3f}, open: {is_open.sum()}/{is_open.numel()}"
            )
            return is_open

        except Exception as e:
            log.warning(f"Could not check drawer state for {self.obj_name}: {e}")
            # Fallback: assume drawer is not open
            return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get debug viewers."""
        return []
