from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim.base import BaseSimHandler
from metasim.utils.setup_util import get_sim_handler_class

_is_first_isaaclab_context = True


class HandlerContext:
    def __init__(self, scenario: ScenarioCfg):
        self.scenario = scenario
        self.handler = get_sim_handler_class(SimType(self.scenario.sim))(scenario)

    def __enter__(self) -> BaseSimHandler:
        if self.scenario.sim == "isaaclab":
            print("Entering isaaclab context")
            global _is_first_isaaclab_context
            if _is_first_isaaclab_context:
                print("First isaaclab context")
                _is_first_isaaclab_context = False
                self.handler.launch()
            else:
                print("Not first isaaclab context")
                try:
                    from omni.isaac.core.utils.stage import create_new_stage
                except ModuleNotFoundError:
                    from isaacsim.core.utils.stage import create_new_stage
                print("Creating new stage")
                create_new_stage()
                self.handler._setup_environment()
                print("Setuped environment")
        else:
            print("Entering other context")
            self.handler.launch()
        print("Launched handler")
        return self.handler

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            log.error(f"Error in SimContext: {exc_value}, {traceback}")
        print("Closing handler")
        self.handler.close()
