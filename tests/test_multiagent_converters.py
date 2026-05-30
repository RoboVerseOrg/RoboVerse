"""Unit tests for the multi-agent (bimanual) dataset converters.

Covers the two get-started converters with no network and no simulator:

- ``9_maniskill_two_robot_stack_cube.py``: ``convert_demo_to_v2`` /
  ``_validate_schema`` on a tiny synthetic ManiSkill-layout ``.h5``.
- ``10_robotwin_aloha_replay.py``: ``bridge_to_v2`` / ``_vector_to_dof`` on a
  tiny synthetic RoboTwin bridge dict.

The example modules start with a digit, so they are loaded by file path.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_GET_STARTED = Path(__file__).resolve().parents[1] / "get_started"


def _load_example(filename: str):
    """Import a ``get_started/NN_*.py`` example module by file path.

    The module is registered in ``sys.modules`` before execution so that the
    ``@dataclass`` / ``@configclass`` definitions in the example can resolve
    their own module (``dataclasses`` looks it up via ``cls.__module__``).
    """
    path = _GET_STARTED / filename
    mod_name = f"_get_started_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Example 9 — ManiSkill TwoRobotStackCube-v1 converter
# ---------------------------------------------------------------------------


def _write_fake_ms_demo(path: Path, ex9, n_frames: int = 4) -> dict:
    """Write a minimal ManiSkill-layout ``.h5`` and return the qpos we stored.

    The articulation row layout mirrors ManiSkill's
    ``hstack([pose.p(3), pose.q(4), lin_vel(3), ang_vel(3), qpos(9), qvel(9)])``.
    """
    h5py = pytest.importorskip("h5py")
    n_dof = len(ex9.PANDA_JOINTS)
    width = 13 + 2 * n_dof  # 31 for a 9-DOF panda
    qpos_by_agent = {}
    with h5py.File(path, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("success", data=np.array([False] * (n_frames - 1) + [True]))
        es = g.create_group("env_states")
        arts = es.create_group("articulations")
        for agent_idx, articulation in enumerate(ex9.AGENT_TO_ARTICULATION.values()):
            row = np.zeros((n_frames, width), dtype=np.float32)
            # Distinct, recognizable pos / quat / qpos per agent and per frame.
            for i in range(n_frames):
                row[i, 0:3] = [0.1 * agent_idx, -0.75 + agent_idx, 0.0]
                row[i, 3:7] = [0.7071, 0.0, 0.0, 0.7071]  # wxyz 90deg about z
                row[i, 13 : 13 + n_dof] = np.arange(n_dof) * 0.01 + 0.1 * i + agent_idx
            arts.create_dataset(articulation, data=row)
            qpos_by_agent[articulation] = row[:, 13 : 13 + n_dof]
        actors = es.create_group("actors")
        for j, name in enumerate(ex9.OBJECT_NAMES):
            arow = np.zeros((n_frames, 13), dtype=np.float32)
            arow[:, 0:3] = [0.3 + 0.1 * j, 0.0, 0.05]
            arow[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
            actors.create_dataset(name, data=arow)
    return qpos_by_agent


@pytest.mark.general
def test_convert_demo_to_v2_is_name_keyed_with_shared_objects(tmp_path):
    """convert_demo_to_v2 yields one entry per agent, shared objects, right qpos."""
    ex9 = _load_example("9_maniskill_two_robot_stack_cube.py")
    h5_path = tmp_path / "trajectory.h5"
    qpos_by_agent = _write_fake_ms_demo(h5_path, ex9, n_frames=4)

    out_path = tmp_path / "two_robot_v2.pkl"
    n_frames = ex9.convert_demo_to_v2(str(h5_path), str(out_path), demo_index=None)
    assert n_frames == 4

    with open(out_path, "rb") as f:
        dataset = pickle.load(f)

    # One entry per agent (keyed by RoboVerse robot name) + metadata.
    for agent_name in ex9.AGENT_TO_ARTICULATION:
        assert agent_name in dataset
    assert dataset["metadata"]["num_agents"] == len(ex9.AGENT_TO_ARTICULATION)

    init = dataset["panda_left"][0]["init_state"]
    # The agent's own robot entry is present...
    assert "panda_left" in init
    # ...and the shared objects ride along in every agent's slice.
    for obj in ex9.OBJECT_NAMES:
        assert obj in init

    # qpos maps onto PANDA_JOINTS in order, frame 0.
    art0 = ex9.AGENT_TO_ARTICULATION["panda_left"]
    expected = qpos_by_agent[art0][0]
    got = init["panda_left"]["dof_pos"]
    assert list(got.keys()) == ex9.PANDA_JOINTS
    np.testing.assert_allclose(list(got.values()), expected, rtol=0, atol=1e-6)

    # wxyz quaternion passed through unchanged.
    np.testing.assert_allclose(init["panda_left"]["rot"], [0.7071, 0.0, 0.0, 0.7071], atol=1e-6)

    # Actions are the next recorded joint pose (state-replay twin): N-1 of them.
    actions = dataset["panda_left"][0]["actions"]
    assert len(actions) == 3
    np.testing.assert_allclose(list(actions[0]["dof_pos_target"].values()), qpos_by_agent[art0][1], rtol=0, atol=1e-6)


@pytest.mark.general
def test_validate_schema_rejects_narrow_articulation(tmp_path):
    """A too-narrow articulation row must raise, not silently mis-slice qpos."""
    h5py = pytest.importorskip("h5py")
    ex9 = _load_example("9_maniskill_two_robot_stack_cube.py")
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("success", data=np.array([True]))
        es = g.create_group("env_states")
        arts = es.create_group("articulations")
        for articulation in ex9.AGENT_TO_ARTICULATION.values():
            arts.create_dataset(articulation, data=np.zeros((1, 10), dtype=np.float32))  # < _MIN_ART_WIDTH
        actors = es.create_group("actors")
        for name in ex9.OBJECT_NAMES:
            actors.create_dataset(name, data=np.zeros((1, 13), dtype=np.float32))
    with pytest.raises(ValueError, match="row width"):
        ex9.convert_demo_to_v2(str(bad), str(tmp_path / "out.pkl"), demo_index=0)


@pytest.mark.general
def test_validate_schema_rejects_missing_agent_key(tmp_path):
    """A missing expected articulation key must raise a clear KeyError."""
    h5py = pytest.importorskip("h5py")
    ex9 = _load_example("9_maniskill_two_robot_stack_cube.py")
    bad = tmp_path / "bad.h5"
    n_dof = len(ex9.PANDA_JOINTS)
    width = 13 + 2 * n_dof
    with h5py.File(bad, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("success", data=np.array([True]))
        es = g.create_group("env_states")
        arts = es.create_group("articulations")
        # Only create the first agent; the second is missing.
        first = next(iter(ex9.AGENT_TO_ARTICULATION.values()))
        arts.create_dataset(first, data=np.zeros((1, width), dtype=np.float32))
        actors = es.create_group("actors")
        for name in ex9.OBJECT_NAMES:
            actors.create_dataset(name, data=np.zeros((1, 13), dtype=np.float32))
    with pytest.raises(KeyError):
        ex9.convert_demo_to_v2(str(bad), str(tmp_path / "out.pkl"), demo_index=0)


# ---------------------------------------------------------------------------
# RoboTwin ALOHA-AgileX bridge converter (roboverse_pack.tasks.robotwin._convert)
# ---------------------------------------------------------------------------


@pytest.mark.general
def test_vector_to_dof_maps_arms_and_scales_grippers():
    """The 14-D bimanual vector lands on the right joints with scaled grippers."""
    from roboverse_pack.tasks.robotwin import _convert

    # L arm = 1..6, L grip = 0.5; R arm = 11..16, R grip = 1.0.
    vec = [1, 2, 3, 4, 5, 6, 0.5, 11, 12, 13, 14, 15, 16, 1.0]
    left_scale, right_scale = (0.0, 0.04), (0.0, 0.08)
    dof = _convert.vector_to_dof(vec, left_scale, right_scale)

    # Left arm joints fl_joint1..6 take vec[0:6].
    for i in range(6):
        assert dof[f"fl_joint{i + 1}"] == float(vec[i])
    # Right arm joints fr_joint1..6 take vec[7:13].
    for i in range(6):
        assert dof[f"fr_joint{i + 1}"] == float(vec[7 + i])

    # Gripper value in [0, 1] maps linearly into the embodiment's scale, mirrored
    # onto both finger joints (joint7 / joint8).
    exp_left = left_scale[0] + 0.5 * (left_scale[1] - left_scale[0])  # 0.02
    exp_right = right_scale[0] + 1.0 * (right_scale[1] - right_scale[0])  # 0.08
    assert dof["fl_joint7"] == dof["fl_joint8"] == pytest.approx(exp_left)
    assert dof["fr_joint7"] == dof["fr_joint8"] == pytest.approx(exp_right)

    # Untouched joints are pinned to zero.
    assert dof["right_wheel"] == 0.0


@pytest.mark.general
def test_vector_to_dof_gripper_endpoints_are_byte_exact():
    """norm=0 -> scale[0] (open) and norm=1 -> scale[1] (closed), no rounding drift."""
    from roboverse_pack.tasks.robotwin import _convert

    ls, rs = (-0.01, 0.045), (-0.02, 0.05)
    open_vec = [0] * 6 + [0.0] + [0] * 6 + [0.0]
    closed_vec = [0] * 6 + [1.0] + [0] * 6 + [1.0]
    dof_open = _convert.vector_to_dof(open_vec, ls, rs)
    dof_closed = _convert.vector_to_dof(closed_vec, ls, rs)
    assert dof_open["fl_joint7"] == ls[0]
    assert dof_open["fr_joint7"] == rs[0]
    assert dof_closed["fl_joint7"] == ls[1]
    assert dof_closed["fr_joint7"] == rs[1]


@pytest.mark.general
def test_vector_to_dof_mimic_defaults_unchanged_and_generalizes():
    """Default mimic keeps joint7==joint8 (byte-identical); a non-unity mimic applies."""
    from roboverse_pack.tasks.robotwin import _convert

    vec = [0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5]
    ls, rs = (0.0, 0.04), (0.0, 0.08)
    # Default (ALOHA unity mimic): both finger joints take the same value.
    dof = _convert.vector_to_dof(vec, ls, rs)
    assert dof["fl_joint7"] == dof["fl_joint8"] == pytest.approx(0.02)
    assert dof["fr_joint7"] == dof["fr_joint8"] == pytest.approx(0.04)
    # Non-unity mimic: joint8 = joint7 * mult + offset (mult/offset not dropped).
    dof2 = _convert.vector_to_dof(vec, ls, rs, left_mimic=(-1.0, 0.01), right_mimic=(0.5, 0.0))
    assert dof2["fl_joint7"] == pytest.approx(0.02)
    assert dof2["fl_joint8"] == pytest.approx(0.02 * -1.0 + 0.01)
    assert dof2["fr_joint8"] == pytest.approx(0.04 * 0.5)


@pytest.mark.general
def test_bridge_to_v2_roundtrip_and_off_by_one(tmp_path):
    """bridge_to_v2 seeds init from frame 0 and emits N-1 actions from frames 1:."""
    from roboverse_pack.tasks.robotwin import _convert

    bridge = {
        "vectors": [
            [0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0.0],
            [1, 2, 3, 4, 5, 6, 1.0, 11, 12, 13, 14, 15, 16, 0.0],
            [2, 3, 4, 5, 6, 7, 0.5, 12, 13, 14, 15, 16, 17, 0.5],
        ],
        "left_gripper_scale": (0.0, 0.04),
        "right_gripper_scale": (0.0, 0.05),
        "init_objects": {"box": {"pos": [0.1, 0.2, 0.74], "rot": [1.0, 0.0, 0.0, 0.0]}},
        "task": "beat_block_hammer",
        "seed": 7,
    }
    out_path = tmp_path / "robotwin_aloha_v2.pkl"
    _convert.bridge_to_v2(bridge, str(out_path))

    with open(out_path, "rb") as f:
        dataset = pickle.load(f)

    assert dataset["metadata"]["num_agents"] == 1
    assert dataset["metadata"]["agents"] == [_convert.ROBOT_NAME]
    assert dataset["metadata"]["source_seed"] == 7
    demo = dataset[_convert.ROBOT_NAME][0]
    # One robot entry plus the static block in the init state.
    assert _convert.ROBOT_NAME in demo["init_state"]
    assert "block" in demo["init_state"]
    # init_state is frame 0; there are N-1 = 2 action steps from frames 1: (off-by-one guard).
    assert demo["init_state"][_convert.ROBOT_NAME]["dof_pos"]["fl_joint1"] == 0.0
    assert len(demo["actions"]) == 2
    assert demo["actions"][0]["dof_pos_target"]["fl_joint1"] == 1.0
    assert demo["actions"][1]["dof_pos_target"]["fl_joint1"] == 2.0
    assert demo["actions"][0]["dof_pos_target"]["fr_joint6"] == 16.0
    # Left gripper fully closed (vec[6]=1.0) -> scale[1].
    assert demo["actions"][0]["dof_pos_target"]["fl_joint7"] == pytest.approx(0.04)


@pytest.mark.general
def test_example10_imports_shared_converter():
    """Example 10 must consume the shared converter, not re-define its own."""
    ex10 = _load_example("10_robotwin_aloha_replay.py")
    from roboverse_pack.tasks.robotwin import _convert

    # The example re-exports the shared symbol (single source of truth).
    assert ex10.bridge_to_v2 is _convert.bridge_to_v2


# ---------------------------------------------------------------------------
# Regression: single-agent loading must stay byte-identical to a list-of-one
# ---------------------------------------------------------------------------


@pytest.mark.general
def test_single_agent_get_traj_unchanged(tmp_path):
    """A single ``RobotCfg`` and a one-element list yield identical trajectories.

    This is the contract ``scripts/advanced/replay_demo.py`` relies on: it passes
    a single ``RobotCfg`` for a one-robot task (the legacy path) and only switches
    to the list form when a task declares multiple robots. The two must agree for
    the single-robot case so existing replays are unaffected.
    """
    from metasim.utils.demo_util import get_traj
    from metasim.utils.demo_util.loader import save_traj_file
    from roboverse_pack.robots import FrankaCfg

    home = {f"panda_joint{i}": 0.0 for i in range(1, 8)}
    home["panda_finger_joint1"] = 0.04
    home["panda_finger_joint2"] = 0.04
    action = {"dof_pos_target": dict(home)}
    dataset = {
        "franka": [
            {
                "init_state": {
                    "franka": {"pos": [0, 0, 0], "rot": [1, 0, 0, 0], "dof_pos": dict(home)},
                    "cube": {"pos": [0.4, 0, 0.05], "rot": [1, 0, 0, 0]},
                },
                "actions": [action, action],
                "states": None,
            }
        ],
        "metadata": {"num_agents": 1},
    }
    path = tmp_path / "single_franka_v2.pkl"
    save_traj_file(dataset, str(path))

    robot = FrankaCfg(name="franka")
    init_single, act_single, _ = get_traj(str(path), robot)
    init_list, act_list, _ = get_traj(str(path), [robot])

    # Legacy single-robot per-step shape: {robot_name: {"dof_pos_target": {...}}}.
    assert list(act_single[0][0]) == ["franka"]
    assert "dof_pos_target" in act_single[0][0]["franka"]
    # Passing a one-element list is byte-identical to the bare RobotCfg.
    assert act_single == act_list
    assert init_single[0]["robots"].keys() == init_list[0]["robots"].keys()
    assert init_single[0]["objects"].keys() == init_list[0]["objects"].keys()
