import math

import torch


def assert_close(a, b, atol=1e-3, message="Consistency Error"):
    if isinstance(a, torch.Tensor):
        assert torch.allclose(a, b, atol=atol), f"a: {a} != b: {b} " + message
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=atol), f"a: {a} != b: {b} " + message
    else:
        raise ValueError(f"Unsupported type: {type(a)}")


def get_test_parameters():
    """Generate test parameters with different num_envs for different simulators."""
    # MuJoCo only supports num_envs=1 due to simulator limitations
    # Other simulators can test with multiple environments
    isaacsim_params = [("isaacsim", num_envs) for num_envs in [1, 2, 4]]
    isaacgym_params = [("isaacgym", num_envs) for num_envs in [1, 2, 4]]
    genesis_params = [("genesis", num_envs) for num_envs in [1, 2, 4]]
    mujoco_params = [("mujoco", 1)]
    sapien3_params = [("sapien3", 1)]
    sapien2_params = [("sapien2", 1)]
    return mujoco_params + isaacsim_params + isaacgym_params + genesis_params + sapien3_params + sapien2_params
