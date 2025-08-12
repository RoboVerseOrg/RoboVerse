from __future__ import annotations

from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class BaseChecker:
    """Base class for all checkers. Checkers are used to check whether the task is executed successfully."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """The code to run when the environment is reset."""
        pass

    def check(self, handler: BaseSimHandler):
        """Check whether the task is executed successfully."""
        import torch

        # log.warning("Checker not implemented, task will never succeed")
        return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get the viewers to be used for debugging the checker."""
        return []
