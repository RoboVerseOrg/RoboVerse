"""Sub-module containing the scenario configuration."""

from __future__ import annotations

import dataclasses
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
            "superdex",
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
        """Patch fields then rerun post-init.

        ``update`` is the public patch boundary — ``gym.make(id, **kwargs)``
        / ``gym.make_vec`` forward their caller's kwargs straight here (see
        ``gym_registration``). Previously every key was applied with a raw
        ``setattr``, so a typo like ``num_env=4`` (for ``num_envs``) or
        ``headles=True`` silently created a dead attribute, left the real
        field at its default, and gave no feedback — the caller's intent was
        dropped without a trace.

        Unknown keys are now **warned and skipped** rather than set as dead
        attributes. A warning (not a hard error) is deliberate: it mirrors
        the existing ``_warn_duplicate_names`` boundary behaviour and avoids
        breaking the several shipped scripts that pass a long-dead ``renderer=``
        kwarg alongside their real ``simulator=`` — those keep working, but
        the dead kwarg now surfaces instead of silently polluting the object.
        """
        valid_fields = {f.name for f in dataclasses.fields(self)}
        unknown = [key for key in kwargs if key not in valid_fields]
        if unknown:
            log.warning(
                f"ScenarioCfg.update() ignoring unknown field(s) {unknown} — "
                f"they are not ScenarioCfg fields and will have no effect "
                f"(check for a typo). Valid fields: {sorted(valid_fields)}."
            )
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
        self.__post_init__()
        return self

    def replace(self, **kwargs):
        """Return a copy with ``kwargs`` applied.

        ``update`` mutates in place and the object is shared by every holder (class-level task
        defaults included); prefer ``replace`` in new code.
        """
        import copy

        return copy.deepcopy(self).update(**kwargs)
