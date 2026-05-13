from __future__ import annotations

import traceback
from typing import Any

from loguru import logger as log

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.utils.setup_util import get_sim_handler_class


class HandlerContext:
    def __init__(self, scenario: ScenarioCfg, simulation_app: Any | None = None):
        self.scenario = scenario
        self.handler = get_sim_handler_class(SimType(self.scenario.simulator))(scenario)
        self.simulation_app = simulation_app

    def __enter__(self) -> BaseSimHandler:
        try:
            if self.scenario.simulator == "isaacsim":
                self.handler.launch(simulation_app=self.simulation_app)
            else:
                self.handler.launch()
        except Exception as e:
            log.error("An error occurred during handler launch: {}", e)
            log.error("Stack trace:\n{}", traceback.format_exc())
            # Always close — including isaacsim. The handler's own
            # ``_owns_simulation_app`` flag decides whether close() does a
            # full Kit-app teardown or only handler-local cleanup, so it
            # is safe to call unconditionally.
            try:
                self.handler.close()
            except Exception:
                log.error("An error occurred during handler cleanup after failed launch.")
                log.error("Stack trace:\n{}", traceback.format_exc())
            raise
        return self.handler

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            log.error("Error in SimContext:")
            log.error("Stack trace:\n{}", traceback.format_exception(exc_type, exc_value, exc_traceback))
        # Always close — skipping it for isaacsim leaked Replicator workers
        # and hydra delegates between runs.
        self.handler.close()
