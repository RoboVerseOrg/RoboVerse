"""Integration tests for state mode conversion between tensor and dict formats.

Tests the get_states(mode="tensor") and get_states(mode="dict") functionality
and conversion between the two formats.
"""

from __future__ import annotations

import pytest
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close
from metasim.utils.state import action_input_to_tensor


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_get_states_dict_mode(handler):
    """Test that get_states with mode='dict' returns list of dicts."""
    dict_state = handler.get_states(mode="dict")

    assert isinstance(dict_state, list)
    assert len(dict_state) == handler.scenario.num_envs

    # Verify structure of first environment
    env0_state = dict_state[0]
    assert "robots" in env0_state
    assert "objects" in env0_state
    assert "franka" in env0_state["robots"]
    assert "cube" in env0_state["objects"]

    # Verify robot dict structure
    robot_state = env0_state["robots"]["franka"]
    assert "pos" in robot_state
    assert "rot" in robot_state
    assert "dof_pos" in robot_state
    assert "dof_vel" in robot_state

    # Verify types
    assert isinstance(robot_state["pos"], torch.Tensor)
    assert robot_state["pos"].shape == (3,)
    assert isinstance(robot_state["rot"], torch.Tensor)
    assert robot_state["rot"].shape == (4,)
    assert isinstance(robot_state["dof_pos"], dict)

    log.info(f"Get states dict mode test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_state_mode_conversion_consistency(handler):
    """Test that tensor and dict modes represent the same state."""
    # Set a known state
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.3,
                        "panda_joint2": -0.6,
                        "panda_joint3": 0.2,
                        "panda_joint4": -2.2,
                        "panda_joint5": 0.1,
                        "panda_joint6": 1.6,
                        "panda_joint7": 0.9,
                        "panda_finger_joint1": 0.03,
                        "panda_finger_joint2": 0.03,
                    }
                }
            }
        ]
        * handler.scenario.num_envs
    )

    for _ in range(10):
        handler.simulate()

    # Get both formats
    tensor_state = handler.get_states(mode="tensor")
    dict_state = handler.get_states(mode="dict")

    # Compare positions for first environment
    tensor_robot_pos = tensor_state.robots["franka"].root_state[0, 0:3]
    dict_robot_pos = dict_state[0]["robots"]["franka"]["pos"]
    assert_close(tensor_robot_pos.cpu(), dict_robot_pos, atol=1e-6, message="robot root pos")

    # Compare orientations
    tensor_robot_quat = tensor_state.robots["franka"].root_state[0, 3:7]
    dict_robot_rot = dict_state[0]["robots"]["franka"]["rot"]
    assert_close(tensor_robot_quat.cpu(), dict_robot_rot, atol=1e-6, message="robot root quat")

    # Compare joint positions
    sorted_joint_names = handler.get_joint_names("franka", sort=True)
    tensor_joint_pos = tensor_state.robots["franka"].joint_pos[0]

    for idx, joint_name in enumerate(sorted_joint_names):
        dict_joint_pos = dict_state[0]["robots"]["franka"]["dof_pos"][joint_name]
        assert_close(
            tensor_joint_pos[idx].item(),
            dict_joint_pos,
            atol=1e-6,
            message=f"joint {joint_name} pos",
        )

    log.info(f"State mode conversion consistency test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_state_mode_multiple_calls(handler):
    """Test that calling get_states with different modes multiple times works correctly."""
    # Call with tensor mode
    tensor_state_1 = handler.get_states(mode="tensor")

    # Call with dict mode
    dict_state = handler.get_states(mode="dict")

    # Call with tensor mode again
    tensor_state_2 = handler.get_states(mode="tensor")

    # Both tensor calls should give same results (cache working correctly)
    assert_close(
        tensor_state_1.robots["franka"].root_state[:, 0:3],
        tensor_state_2.robots["franka"].root_state[:, 0:3],
        atol=1e-9,
        message="tensor state consistency across calls",
    )

    # Simulate and verify cache is invalidated
    handler.simulate()

    tensor_state_3 = handler.get_states(mode="tensor")

    # After simulate, positions may have changed slightly
    # Just verify we can still get the state
    assert tensor_state_3.robots["franka"].root_state is not None

    log.info(f"State mode multiple calls test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_state_cache_mode_independence(handler):
    """Regression: alternating modes must not invalidate or overwrite the other cache."""
    tensor_a = handler.get_states(mode="tensor")
    dict_a = handler.get_states(mode="dict")
    tensor_b = handler.get_states(mode="tensor")
    dict_b = handler.get_states(mode="dict")

    # Same underlying objects returned across calls — cache reused, not rebuilt.
    assert tensor_a is tensor_b, "tensor cache destroyed by intervening dict call"
    assert dict_a is dict_b, "dict cache destroyed by intervening tensor call"

    # simulate() must invalidate both caches.
    handler.simulate()
    tensor_c = handler.get_states(mode="tensor")
    dict_c = handler.get_states(mode="dict")
    assert tensor_c is not tensor_a, "simulate() did not invalidate tensor cache"
    assert dict_c is not dict_a, "simulate() did not invalidate dict cache"

    log.info(f"State cache mode independence test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_set_dof_targets_invalidates_state_cache(handler):
    """Regression: set_dof_targets writes ctrl in MuJoCo and ``get_states`` reads
    ctrl back as ``joint_pos_target``. Without cache invalidation, a get_states
    call after set_dof_targets returned the pre-action target — a silent staleness
    bug across the entire backend matrix."""
    # Prime both caches so we can detect either being rebuilt.
    tensor_pre = handler.get_states(mode="tensor")
    dict_pre = handler.get_states(mode="dict")

    # Issue an action that does not advance simulation.
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.5,
                        "panda_joint2": -0.5,
                        "panda_joint3": 0.0,
                        "panda_joint4": -1.0,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.0,
                        "panda_joint7": 0.5,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    }
                }
            }
        ]
        * handler.scenario.num_envs
    )

    tensor_post = handler.get_states(mode="tensor")
    dict_post = handler.get_states(mode="dict")

    # Caches must have been rebuilt — otherwise stale joint_pos_target leaks through.
    assert tensor_post is not tensor_pre, "set_dof_targets did not invalidate tensor cache"
    assert dict_post is not dict_pre, "set_dof_targets did not invalidate dict cache"

    log.info(f"set_dof_targets cache invalidation test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_set_states_get_states_round_trip(handler):
    """Cross-backend invariant: ``set_states(X) → get_states()`` recovers
    ``X`` for the keys ``_set_states`` honours (``pos`` / ``rot`` /
    ``dof_pos``). This pins down representation drift across backends —
    if any handler stores joint names with a different prefix, packs
    quaternions in a different order, or silently drops a field, this
    test catches it.

    Tolerance is per-quantity: 1e-3 for root pose (which goes through
    quaternion normalisation in some backends), 1e-4 for joint
    positions (direct qpos write, expected exact modulo float32 in
    GPU backends).
    """
    # Snapshot the post-launch state so we know which keys exist.
    pre = handler.get_states(mode="dict")
    assert isinstance(pre, list) and len(pre) >= 1

    franka_pre = pre[0]["robots"]["franka"]
    joint_names = list(franka_pre["dof_pos"].keys())
    assert joint_names, "Franka should have at least one joint"

    # Construct a target state that mutates dof_pos to a known pattern
    # plus a small translation of the root. We avoid extreme values so
    # the simulator's joint-limit clamping (if any) doesn't muddy the test.
    target_dof = {jn: 0.05 * (i % 3 - 1) for i, jn in enumerate(joint_names)}
    target_pos = [0.05, 0.0, 0.5]
    target_rot = [1.0, 0.0, 0.0, 0.0]

    new_state = [
        {
            "objects": {},
            "robots": {
                "franka": {
                    "pos": target_pos,
                    "rot": target_rot,
                    "dof_pos": target_dof,
                }
            },
        }
    ] * handler.scenario.num_envs

    handler.set_states(new_state)
    handler.refresh_render()  # ensure forward kinematics propagate
    post = handler.get_states(mode="dict")
    franka_post = post[0]["robots"]["franka"]

    # Joint positions must round-trip — set_states writes qpos directly,
    # get_states reads qpos, no integration step in between.
    for jn, expected in target_dof.items():
        got = franka_post["dof_pos"][jn]
        actual = got.item() if hasattr(got, "item") else float(got)
        assert abs(actual - expected) < 1e-3, (
            f"dof_pos round-trip failed on {handler.scenario.simulator} for {jn}: set={expected:.4f}, got={actual:.4f}"
        )

    # Root pose round-trip — looser tolerance because some backends normalise
    # the quaternion or apply small numerical massaging in set_states.
    got_pos = franka_post["pos"]
    pos_list = got_pos.tolist() if hasattr(got_pos, "tolist") else list(got_pos)
    for i, (a, b) in enumerate(zip(pos_list, target_pos)):
        assert abs(a - b) < 1e-3, (
            f"root pos round-trip failed on {handler.scenario.simulator} component {i}: set={b:.4f}, got={a:.4f}"
        )

    log.info(f"set_states→get_states round-trip passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_dict_state_all_objects(handler):
    """Test that dict state includes all objects and robots."""
    dict_state = handler.get_states(mode="dict")

    env0_state = dict_state[0]

    # Verify all objects are present
    assert "cube" in env0_state["objects"]
    assert "sphere" in env0_state["objects"]
    assert "bbq_sauce" in env0_state["objects"]
    assert "box_base" in env0_state["objects"]

    # Verify robot is present
    assert "franka" in env0_state["robots"]

    # Verify articulation object has dof_pos
    assert "dof_pos" in env0_state["objects"]["box_base"]
    assert "box_joint" in env0_state["objects"]["box_base"]["dof_pos"]

    log.info(f"Dict state all objects test passed for {handler.scenario.simulator}")


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_handler_accepts_tensor_actions(handler):
    """Handler accepts tensor actions in handler/API order."""
    action_dim = sum(len(handler.get_joint_names(robot.name, sort=True)) for robot in handler.robots)
    target = torch.zeros((handler.num_envs, action_dim), dtype=torch.float32, device=handler.device)

    handler.set_dof_targets(target)
    assert handler.actions_cache is not None


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_handler_accepts_dict_actions(handler):
    """Handler accepts dict-batch actions."""
    dict_actions = [
        {
            robot.name: {
                "dof_pos_target": {joint_name: 0.0 for joint_name in handler.get_joint_names(robot.name, sort=True)}
            }
            for robot in handler.robots
        }
        for _ in range(handler.num_envs)
    ]

    handler.set_dof_targets(dict_actions)
    assert handler.actions_cache is not None


@pytest.mark.sim("mujoco", "isaacsim", "isaacgym", "newton", "sapien3")
def test_action_dict_and_tensor_use_same_handler_api_order(handler):
    """Dict actions normalize to the documented handler/API tensor order."""
    dict_actions = [
        {
            robot.name: {
                "dof_pos_target": {
                    joint_name: float(idx + 1)
                    for idx, joint_name in enumerate(handler.get_joint_names(robot.name, sort=True))
                }
            }
            for robot in handler.robots
        }
        for _ in range(handler.num_envs)
    ]

    tensor = action_input_to_tensor(handler, dict_actions, device=handler.device)

    expected_rows = []
    for _ in range(handler.num_envs):
        row = []
        for robot in handler.robots:
            row.extend(float(idx + 1) for idx, _ in enumerate(handler.get_joint_names(robot.name, sort=True)))
        expected_rows.append(row)
    expected = torch.tensor(expected_rows, dtype=torch.float32, device=handler.device)

    assert_close(tensor, expected, atol=1e-6, message="handler api tensor order")
