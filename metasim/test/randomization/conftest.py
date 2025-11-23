"""Register randomization suite with the shared handler utilities."""

from __future__ import annotations

import pytest

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite

_ALLOWED_SIMS = {"isaacsim"}

# Tests that modify USD materials and can invalidate PhysX views for subsequent tests.
# These should run LAST within each handler to avoid contaminating other tests.
_CONTAMINATING_TEST_NAMES = {"material_pbr", "material_mdl"}


def get_randomization_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create a standard scenario configuration for randomization tests."""
    if sim not in _ALLOWED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for randomization tests")

    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
    from roboverse_pack.robots.franka_cfg import FrankaCfg

    return ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.4, -0.6, 0.05],
            ),
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.5, 0.0, 0.5],
            ),
        ],
        lights=[
            DiskLightCfg(
                name="test_light",
                intensity=20000.0,
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, 4.5),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        robots=[FrankaCfg()],
        cameras=[
            PinholeCameraCfg(
                name="test_camera",
                width=1024,
                height=1024,
                pos=(2.0, -2.0, 2.0),
                look_at=(0.0, 0.0, 0.05),
            )
        ],
    )


def _get_handler_key(item: pytest.Item) -> tuple[str, ...] | None:
    """Extract handler parameters to group tests by shared handler."""
    if not hasattr(item, "callspec"):
        return None
    params = getattr(item.callspec, "params", {})
    handler_param = params.get("handler")
    if not isinstance(handler_param, dict):
        return None
    sim = handler_param.get("sim", "")
    num_envs = handler_param.get("num_envs", 0)
    return (sim, str(num_envs))


def _is_contaminating_test(item: pytest.Item) -> bool:
    """Check if a test function is known to contaminate PhysX views."""
    # Get the test_func parameter from parametrized tests
    if hasattr(item, "callspec"):
        params = getattr(item.callspec, "params", {})
        test_func = params.get("test_func")
        if test_func is not None and hasattr(test_func, "__name__"):
            return test_func.__name__ in _CONTAMINATING_TEST_NAMES
    # Fallback: check item name
    for name in _CONTAMINATING_TEST_NAMES:
        if name in item.name:
            return True
    return False


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder tests so contaminating tests run last within each handler group.

    Tests that modify USD materials (material_pbr, material_mdl) can invalidate
    PhysX simulation views. By running them last for each handler, we prevent
    them from affecting other tests that rely on valid PhysX views.
    """
    # Only process tests from this module
    randomization_items = [item for item in items if "metasim/test/randomization" in str(item.fspath)]
    if not randomization_items:
        return

    # Group tests by handler key
    handler_groups: dict[tuple[str, ...] | None, list[pytest.Item]] = {}
    for item in randomization_items:
        key = _get_handler_key(item)
        if key not in handler_groups:
            handler_groups[key] = []
        handler_groups[key].append(item)

    # Within each group, sort so contaminating tests come last
    reordered: list[pytest.Item] = []
    for key in sorted(handler_groups.keys(), key=lambda k: k or ("", "")):
        group = handler_groups[key]
        safe_tests = [item for item in group if not _is_contaminating_test(item)]
        contaminating_tests = [item for item in group if _is_contaminating_test(item)]
        reordered.extend(safe_tests)
        reordered.extend(contaminating_tests)

    # Replace items in the original list
    # First, find indices of randomization items
    other_items = [item for item in items if item not in randomization_items]
    items.clear()
    items.extend(other_items)
    items.extend(reordered)


# Register this suite with the shared handler machinery in metasim/test/conftest.py
register_shared_suite("metasim.test.randomization", get_randomization_scenario)
