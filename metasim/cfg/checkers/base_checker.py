from __future__ import annotations

from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class BaseChecker:
    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        pass

    def check(self, handler: BaseSimHandler):
        import torch

        # log.warning("Checker not implemented, task will never succeed")
        return torch.zeros(handler.num_envs, dtype=torch.bool)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []
