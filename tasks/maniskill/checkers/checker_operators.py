## ruff: noqa: D102

from __future__ import annotations

from dataclasses import MISSING

import torch

from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass

from .base_checker import BaseChecker

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class AndOp(BaseChecker):
    """Combine multiple checkers with a logical AND operation."""

    checkers: list[BaseChecker] = MISSING

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        for checker in self.checkers:
            checker.reset(handler, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        success = torch.ones(handler.num_envs, dtype=torch.bool, device=handler.device)
        for checker in self.checkers:
            success = success & checker.check(handler)
        return success

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        viewers = []
        for checker in self.checkers:
            viewers += checker.get_debug_viewers()
        return viewers


@configclass
class OrOp(BaseChecker):
    """Combine multiple checkers with a logical OR operation."""

    checkers: list[BaseChecker] = MISSING

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        for checker in self.checkers:
            checker.reset(handler, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        success = torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)
        for checker in self.checkers:
            success = success | checker.check(handler)
        return success

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        viewers = []
        for checker in self.checkers:
            viewers += checker.get_debug_viewers()
        return viewers


@configclass
class NotOp(BaseChecker):
    """Negate the result of a checker."""

    checker: BaseChecker = MISSING

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        self.checker.reset(handler, env_ids=env_ids)

    def check(self, handler: BaseSimHandler) -> torch.BoolTensor:
        return ~self.checker.check(handler)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return self.checker.get_debug_viewers()
