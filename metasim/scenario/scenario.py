"""Sub-module containing the scenario configuration."""

from __future__ import annotations

from typing import Literal

from metasim.utils.configclass import configclass
from metasim.utils.hf_util import FileDownloader
from metasim.utils.setup_util import get_robot, get_scene

from .cameras import BaseCameraCfg
from .lights import BaseLightCfg, DistantLightCfg
from .objects import BaseObjCfg
from .render import RenderCfg
from .robot import RobotCfg
from .scene import GSSceneCfg, SceneCfg
from .simulator_params import SimParamCfg


@configclass
class ScenarioCfg:
    """Scenario configuration."""

    # assets
    scene: SceneCfg | None = None
    robots: list[RobotCfg] = []
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = []
    gs_scene: GSSceneCfg | None = None

    # runtime
    render: RenderCfg = RenderCfg()
    sim_params: SimParamCfg = SimParamCfg()
    simulator: (
        Literal[
            "isaaclab",
            "isaacgym",
            "sapien2",
            "sapien3",
            "genesis",
            "pybullet",
            "mujoco",
        ]
        | None
    ) = None

    # misc
    num_envs: int = 1
    headless: bool = False
    env_spacing: float = 1.0
    decimation: int = 25
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def __post_init__(self) -> None:
        """Resolve strings & fetch assets; skip until `simulator` is set."""
        # if self.simulator is None:  # defer init until user specifies simulator
        #     return

        for i, robot in enumerate(self.robots):
            if isinstance(robot, str):
                self.robots[i] = get_robot(robot)

        if isinstance(self.scene, str):
            self.scene = get_scene(self.scene)

        # FileDownloader(self).do_it()  # download any external assets

    def check_assets(self):
        """Check if all assets are available."""
        FileDownloader(self).do_it()  # download any external assets

    def update(self, **kwargs):
        """Patch fields then rerun post-init."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__post_init__()
        return self
