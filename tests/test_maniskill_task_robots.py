"""Every registered ManiSkill task must run with a robot arm.

Each config-port leaf declares ``scenario = ScenarioCfg(objects=[...])``, which *replaces* the
base class's scenario wholesale — and ``ScenarioCfg.robots`` defaults to ``[]``. Only a handful
of leaves (pick_cube, the ``*_dense`` variants, ...) passed ``robots=["franka"]``, so ~2.6k task
classes (every pick_single_egad / peg_insertion_side / pick_single_ycb object variant, plus
plug_charger, stack_pyramid, pull_cube_tool, place_sphere, poke_cube) shipped with an empty robot
list. Constructing one died in ``ManiskillBaseTask._get_initial_states`` with a bare
``IndexError: list index out of range`` from ``self.scenario.robots[0]``.

``ManiskillBaseTask.default_robots`` now supplies the family's arm when a leaf's scenario names
none. These tests pin both halves of that contract:

- every registered ManiSkill task names a robot, directly or via the family default;
- constructing a leaf from each affected family actually reaches ``get_traj`` with a franka
  RobotCfg, instead of raising IndexError.

Backend-free: the handler and the trajectory loader are stubbed, so nothing touches a simulator
or the HuggingFace asset store.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import TASK_REGISTRY, get_task_class
from roboverse_pack.tasks.maniskill.maniskill_base import ManiskillBaseTask

# One leaf per family that shipped an empty robot list, plus pick_cube as the control
# (it always declared robots=["franka"]). peg_insertion_side_81 is included because it
# additionally had `scenario = ScenarioCfg` (the class, not an instance) with a stray
# top-level `objects = [...]`, so it could not construct at all.
_AFFECTED_FAMILIES = [
    "maniskill.peg_insertion_side_363",
    "maniskill.peg_insertion_side_81",
    "maniskill.pick_single_egad_a100",
    "maniskill.pick_single_ycb_lego_duplo",
    "maniskill.place_sphere",
    "maniskill.plug_charger",
    "maniskill.poke_cube",
    "maniskill.pull_cube_tool",
    "maniskill.stack_pyramid",
]
_CONTROL = "maniskill.pick_cube"


def _maniskill_task_classes() -> dict[str, type[ManiskillBaseTask]]:
    """Registered ManiSkill config-port tasks, de-duplicated across aliases."""
    get_task_class(_CONTROL)  # force task discovery to populate the registry
    return {
        cls.__name__: cls
        for cls in TASK_REGISTRY.values()
        if isinstance(cls, type) and issubclass(cls, ManiskillBaseTask) and cls is not ManiskillBaseTask
    }


@pytest.mark.general
def test_every_registered_maniskill_task_names_a_robot():
    classes = _maniskill_task_classes()
    assert len(classes) > 2000, f"expected the full ManiSkill registry, only saw {len(classes)} classes"

    robotless = [
        name
        for name, cls in classes.items()
        if not getattr(cls.scenario, "robots", None) and not getattr(cls, "default_robots", None)
    ]
    assert not robotless, (
        f"{len(robotless)} ManiSkill task(s) name no robot and have no family default, so"
        f" _get_initial_states() will IndexError on scenario.robots[0]:"
        f" {sorted(robotless)[:5]}{' ...' if len(robotless) > 5 else ''}"
    )


@pytest.mark.general
def test_every_maniskill_scenario_is_a_scenariocfg_instance():
    """``scenario = ScenarioCfg`` (the class) is not a scenario — the task can never construct."""
    bad = {
        name: cls.scenario
        for name, cls in _maniskill_task_classes().items()
        if cls.scenario is not None and not isinstance(cls.scenario, ScenarioCfg)
    }
    assert not bad, f"task scenario must be a ScenarioCfg instance, not a class/other: {bad}"


@pytest.mark.general
@pytest.mark.parametrize("task_id", [*_AFFECTED_FAMILIES, _CONTROL])
def test_maniskill_task_constructs_with_a_franka(task_id, monkeypatch):
    """The real __init__ path must reach get_traj with a franka RobotCfg (was: IndexError)."""
    seen: dict[str, object] = {}

    def fake_get_traj(traj_filepath, robot, handler):
        seen["robot"] = robot
        return [{"objects": {}, "robots": {}}], None, None

    def fake_instantiate_env(self, scenario):
        self.handler = MagicMock()
        self.handler.device = "cpu"

    monkeypatch.setattr("roboverse_pack.tasks.maniskill.maniskill_base.get_traj", fake_get_traj)
    monkeypatch.setattr("roboverse_pack.tasks.maniskill.maniskill_base.check_and_download_single", lambda *a: None)
    monkeypatch.setattr("metasim.task.base.check_and_download_single", lambda *a: None)
    monkeypatch.setattr(BaseTaskEnv, "_instantiate_env", fake_instantiate_env)

    cls = get_task_class(task_id)
    # deepcopy: the class-level ScenarioCfg is shared, don't let one test mutate it for the next.
    task = cls(copy.deepcopy(cls.scenario))

    assert task.scenario.robots, f"{task_id} constructed with no robot"
    assert task.scenario.robots[0].name == "franka"
    assert seen["robot"] is task.scenario.robots[0], f"{task_id} did not load its demo traj for the robot"
