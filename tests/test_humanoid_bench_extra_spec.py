"""Regression: humanoid_bench tasks must override _extra_spec, not shadow extra_spec.

``BaseTaskEnv.extra_spec`` is a *property* (returns ``self._extra_spec()``) that the
base reads as ``queries = self.extra_spec`` to register optional site queries. The
sit/slide/stair tasks defined a plain method ``extra_spec`` instead of the
``_extra_spec`` hook, which shadowed the property — so ``self.extra_spec`` was a
bound method (never called), the SitePos queries were never registered, and the
rewards' ``states.extras["head_pos"]`` / ``["imu_pos"]`` lookups would KeyError.
crawl.py used the correct ``_extra_spec`` name. Backend-free: a pure class-level
check (no construction needed).
"""

from __future__ import annotations

import importlib
import inspect

import pytest


@pytest.mark.general
@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("roboverse_pack.tasks.humanoid_bench.sit", "SitEnv"),
        ("roboverse_pack.tasks.humanoid_bench.slide", "SlideEnv"),
        ("roboverse_pack.tasks.humanoid_bench.stair", "StairEnv"),
    ],
)
def test_uses_underscore_extra_spec_hook(module_name, class_name):
    cls = getattr(importlib.import_module(module_name), class_name)

    # The subclass must override the _extra_spec hook, not the extra_spec property.
    assert "_extra_spec" in cls.__dict__, f"{class_name} must override _extra_spec"
    assert "extra_spec" not in cls.__dict__, (
        f"{class_name} defines a plain extra_spec method, shadowing the base property; rename it to _extra_spec"
    )
    # extra_spec must still resolve to the base property (so the base reads the dict).
    assert isinstance(inspect.getattr_static(cls, "extra_spec"), property)
