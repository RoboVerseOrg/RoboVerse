import math

import pytest
import torch
from loguru import logger as log


def assert_close(a, b, atol=1e-3, message="Consistency Error"):
    if isinstance(a, torch.Tensor):
        assert torch.allclose(a, b, atol=atol), f"a: {a} != b: {b} " + message
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=atol), f"a: {a} != b: {b} " + message
    else:
        raise ValueError(f"Unsupported type: {type(a)}")


def get_test_parameters():
    """Generate test parameters with different num_envs for different simulators.

    Note: MuJoCo, MJX, SAPIEN2, and SAPIEN3 are limited to num_envs=1 in tests due to
    current test setup constraints (single-environment physics data access patterns).
    IsaacGym, IsaacSim, and Genesis support multiple parallel environments.
    """
    isaacsim_params = [("isaacsim", num_envs) for num_envs in [1, 2, 4]]
    isaacgym_params = [("isaacgym", num_envs) for num_envs in [1, 2, 4]]
    genesis_params = [("genesis", num_envs) for num_envs in [1, 2, 4]]
    mujoco_params = [("mujoco", 1)]
    mjx_params = [("mjx", 1)]
    sapien3_params = [("sapien3", 1)]
    sapien2_params = [("sapien2", 1)]
    return (
        mujoco_params
        + mjx_params
        + isaacsim_params
        + isaacgym_params
        + genesis_params
        + sapien3_params
        + sapien2_params
    )


@pytest.fixture(scope="session")
def isaacsim_app(request):
    """
    Create an IsaacSim app if any test case in the session uses "isaacsim". Otherwise, do nothing.
    """
    needs_isaacsim = False

    # Get the session object to access collected tests
    session = request.session
    if hasattr(session, "items"):
        for item in session.items:
            # Check if this test has callspec (parametrized) and uses "isaacsim"
            if hasattr(item, "callspec") and "sim" in item.callspec.params:
                if item.callspec.params["sim"] == "isaacsim":
                    needs_isaacsim = True
                    break

    if needs_isaacsim:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
        yield app
        # NOTE: Don't call app.close(), otherwise pytest summary will be skipped!
    else:
        yield None


@pytest.fixture(scope="function", autouse=True)
def isaacsim_context(request):
    """
    Create a new IsaacSim stage if the current test case uses "isaacsim". Otherwise, do nothing.
    """
    sim = request.node.callspec.params.get("sim")

    if sim == "isaacsim":
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils

        log.debug("Creating new stage")
        stage_utils.create_new_stage()
        log.debug("New stage created")
        sim_cfg = sim_utils.SimulationCfg()  # TODO: pass parameters from scenario cfg
        sim_context = sim_utils.SimulationContext(sim_cfg)
        sim_context._app_control_on_stop_handle = None
        yield sim_context
        log.debug("Clearing simulation context")
        sim_context.clear_all_callbacks()
        sim_context.clear_instance()
        log.debug("Simulation context cleared")
    else:
        yield None
