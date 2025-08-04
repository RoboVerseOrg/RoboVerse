"""Sub-module containing the scenario configuration."""

from __future__ import annotations

from typing import Literal

from metasim.utils.configclass import configclass
from metasim.utils.hf_util import FileDownloader
from metasim.utils.setup_util import get_robot, get_scene

from .lights import BaseLightCfg, DistantLightCfg
from .objects import BaseObjCfg
from .render import RenderCfg
from .robots.base_robot_cfg import BaseRobotCfg
from .scenes.base_scene_cfg import SceneCfg
from .cameras import BaseCameraCfg
from .simulator_params import SimParamCfg


@configclass
class ScenarioCfg:
    """Scenario configuration."""

    scene: SceneCfg | None = None
    robots: list[BaseRobotCfg] = []
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = []


    def __post_init__(self):
        """Post-initialization configuration."""
        ### Parse task and robot
        for i, robot in enumerate(self.robots):
            if isinstance(robot, str):
                self.robots[i] = get_robot(robot)
        if isinstance(self.scene, str):
            self.scene = get_scene(self.scene)

        FileDownloader(self).do_it()

    # @configclass
    # class SimulationCfg:
    #     """Simulation configuration."""
    # scenario: ScenarioCfg = ScenarioCfg()

    render: RenderCfg = RenderCfg()
    sim_params: SimParamCfg = SimParamCfg()
    simulator: Literal["isaaclab", "isaacgym", "sapien2", "sapien3", "genesis", "pybullet", "mujoco"] = "isaaclab"
    ## Others
    num_envs: int = 1
    headless: bool = False
    """environment spacing for parallel environments"""
    env_spacing: float = 1.0
    decimation: int = 25

    # trajectory: TrajectoryCfg | None = None
