"""Sub-module containing the scenario configuration."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from loguru import logger as log

from metasim.utils.configclass import configclass
from metasim.utils.hf_util import FileDownloader
from metasim.utils.setup_util import get_ground, get_robot, get_scene

from .cameras import BaseCameraCfg
from .grounds import GroundCfg
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
    ground: GroundCfg | None = None
    add_default_ground: bool = True
    """When `ground` is None, attach a backend-default ground plane. Set False
    to opt out for scenes whose MJCF already provides its own floor (e.g.
    dm_control-style cartpole, mjlab playgrounds)."""

    # runtime
    render: RenderCfg = RenderCfg()
    sim_params: SimParamCfg = SimParamCfg()
    # Must list every value in ``metasim.constants.SimType``. Keep alphabetised.
    simulator: (
        Literal[
            "blender",
            "genesis",
            "isaacgym",
            "isaacsim",
            "mjx",
            "mujoco",
            "newton",
            "pybullet",
            "pyrep",
            "sapien2",
            "sapien3",
        ]
        | None
    ) = None

    # misc
    num_envs: int = 1
    headless: bool = False
    env_spacing: float = 1.0
    decimation: int = 15
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

        if isinstance(self.ground, str):
            self.ground = get_ground(self.ground)

        self._warn_duplicate_names()

    def _warn_duplicate_names(self) -> None:
        """Surface duplicate robot/object names that silently break at launch.

        ``object_dict = {obj.name: obj for obj in self.objects + self.robots}``
        in every handler's ``__init__`` collapses duplicates so the
        second entity silently overrides the first. Different backends
        then disagree about what's actually in the scene (MuJoCo's MJCF
        composer errors on duplicate body names; Sapien3 silently picks
        the first; Newton's behaviour depends on the path). Catch this
        at scenario construction so the user sees the issue before any
        sim is launched.
        """
        names: list[tuple[str, str]] = []  # (kind, name)
        for robot in self.robots:
            name = getattr(robot, "name", None)
            if name:
                names.append(("robot", name))
        for obj in self.objects:
            name = getattr(obj, "name", None)
            if name:
                names.append(("object", name))

        # Count name occurrences across both robots and objects.
        all_names = [name for _, name in names]
        duplicates = [name for name, count in Counter(all_names).items() if count > 1]
        if not duplicates:
            return
        for dup in duplicates:
            kinds = [kind for kind, name in names if name == dup]
            log.warning(
                f"ScenarioCfg has duplicate entity name '{dup}' (used by "
                f"{len(kinds)} entities: {kinds}). Backends silently collapse "
                f"these into the last entry, so set_states / set_dof_targets / "
                f"get_states will all reference the wrong entity. Give each "
                f"robot and object a unique name."
            )

    def check_assets(self):
        """Check if all assets are available."""
        FileDownloader(self).do_it()  # download any external assets

    def update(self, **kwargs):
        """Patch fields then rerun post-init."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__post_init__()
        return self
