"""RoboVerse's benchmark suites are discoverable and name only registered, buildable tasks.

Whether a task's demos replay is a property of the built env on a backend (some tasks choose their
demo file in ``__init__``), so it is gated by ``python -m metasim bench health``, not here.
"""

from __future__ import annotations

import pytest

from metasim.bench import get_suite, list_suites
from metasim.task.registry import get_task_class, list_tasks


def test_tabletop_suite_is_discovered_through_the_package_roots():
    assert "tabletop-mujoco" in list_suites()
    suite = get_suite("tabletop-mujoco")
    assert suite.simulator == "mujoco"
    assert suite.protocol.demo_budget_factor == 3.0 and suite.protocol.episodes == 50


@pytest.mark.parametrize("name", list_suites())
def test_every_suite_task_is_registered(name):
    registered = set(list_tasks())
    for split, entry in get_suite(name).entries():
        assert entry.task in registered, f"{name}/{split}: {entry.task} is not a registered task"
        cls = get_task_class(entry.task)
        assert getattr(cls, "scenario", None) is not None, f"{entry.task} has no class-level scenario"


@pytest.mark.parametrize("name", list_suites())
def test_every_suite_task_supports_a_heldout_language_evaluation(name):
    from metasim.bench import EvalProtocol, PromptBank, plan_instances

    for _, entry in get_suite(name).entries():
        bank = getattr(get_task_class(entry.task), "prompt_bank", None)
        assert isinstance(bank, PromptBank), f"{entry.task} declares no prompt_bank"
        assert len(bank.train) >= 8 and len(bank.heldout) >= 4, f"{entry.task}: too few paraphrases"
        plan = plan_instances([], EvalProtocol(episodes=20, prompt_split="heldout"), prompts=bank, task=entry.task)
        assert {p.prompt for p in plan} <= set(bank.heldout)
