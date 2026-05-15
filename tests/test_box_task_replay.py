"""Integration tests for the box_task replay task and replay CLI."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.general
def test_task_is_registered_under_both_names():
    """``box_task.replay`` is the canonical name; ``box_task`` is the alias."""
    import roboverse_pack.tasks.box_task  # noqa: F401 — triggers @register_task
    from metasim.task.registry import TASK_REGISTRY

    assert "box_task.replay" in TASK_REGISTRY
    assert "box_task" in TASK_REGISTRY
    assert TASK_REGISTRY["box_task.replay"] is TASK_REGISTRY["box_task"]


@pytest.mark.general
def test_task_uses_canonical_robot_name():
    """The scenario must reference the canonical bimanual robot name —
    that's what the converted trajectory pkl keys by."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    robot = BoxTaskReplayEnv.scenario.robots[0]
    assert robot.name == "openarm_bimanual_wuji"


@pytest.mark.general
def test_traj_path_exists_and_is_v2():
    """The committed integration path must point at an existing v2 pkl."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    traj = REPO_ROOT / BoxTaskReplayEnv.traj_filepath
    if not traj.exists():
        pytest.skip(f"trajectory not present locally: {traj}")
    assert traj.suffix == ".pkl"
    assert "_v2" in traj.name


@pytest.mark.general
def test_converted_traj_uses_canonical_names_throughout():
    """The pkl produced by scripts/convert_box_task_legacy_traj.py must:

    - top-level key matches the robot's ``name``
    - inner robot-state dofs use ``{side}_finger{i}_joint{j}`` (no
      ``hand_finger`` leftovers)
    - action dicts use the same finger naming
    """
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    traj = REPO_ROOT / BoxTaskReplayEnv.traj_filepath
    if not traj.exists():
        pytest.skip(f"trajectory not present locally: {traj}")

    with traj.open("rb") as f:
        data = pickle.load(f)

    assert list(data.keys()) == ["openarm_bimanual_wuji"]
    ep = data["openarm_bimanual_wuji"][0]
    assert {"init_state", "states", "actions"}.issubset(ep.keys())
    assert len(ep["states"]) == 849

    # No legacy ``hand_finger`` keys anywhere in a sampled frame.
    sample_state = ep["states"][0]["openarm_bimanual_wuji"]
    sample_dof = sample_state["dof_pos"]
    assert not any("hand_finger" in k for k in sample_dof)
    # All 40 finger joints are present in current form.
    expected = {f"{side}_finger{i}_joint{j}" for side in ("left", "right") for i in range(1, 6) for j in range(1, 5)}
    assert expected.issubset(sample_dof.keys())
    # Actions also remapped.
    assert not any("hand_finger" in k for k in ep["actions"][0]["dof_pos_target"])


@pytest.mark.general
def test_traj_loads_via_get_traj_without_runtime_remap():
    """``get_traj`` should consume the converted pkl directly — no
    per-frame fixup hook on the task class."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    traj = REPO_ROOT / BoxTaskReplayEnv.traj_filepath
    if not traj.exists():
        pytest.skip(f"trajectory not present locally: {traj}")

    from metasim.utils.demo_util import get_traj

    robot = BoxTaskReplayEnv.scenario.robots[0]
    _init, _actions, states = get_traj(str(traj), robot)
    assert len(states) == 1
    assert len(states[0]) == 849
    frame0 = states[0][0]
    assert "openarm_bimanual_wuji" in frame0["robots"]

    # The task should NOT need to define a ``prepare_replay_state``
    # hook — the trajectory is already in canonical form.
    assert not hasattr(BoxTaskReplayEnv, "prepare_replay_state")


@pytest.mark.general
def test_bundled_object_usds_are_present():
    """Asset USDs must be on disk so isaacsim can load the scenario."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    asset_paths = [o.usd_path for o in BoxTaskReplayEnv.scenario.objects if getattr(o, "usd_path", None)]
    missing = [p for p in asset_paths if not (REPO_ROOT / p).exists()]
    if missing:
        pytest.skip(f"asset files not present locally: {missing}")
    assert len(asset_paths) == 3


@pytest.mark.general
def test_replay_cli_module_loads_and_helpers_work():
    """The CLI is loadable as a module and its pure helpers behave."""
    script = REPO_ROOT / "get_started" / "replay_multi_scene_render.py"
    spec = importlib.util.spec_from_file_location("rsmsr", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._parse_scenes("0021,022,usd_0024,kujiale_scene_0031") == [
        "kujiale_scene_0021",
        "kujiale_scene_0022",
        "kujiale_scene_0024",
        "kujiale_scene_0031",
    ]
    with pytest.raises(ValueError):
        mod._parse_scenes("not-a-scene")


@pytest.mark.general
def test_replay_cli_reads_traj_length():
    """The coordinator infers source frame count from the pkl."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    traj = REPO_ROOT / BoxTaskReplayEnv.traj_filepath
    if not traj.exists():
        pytest.skip(f"trajectory not present locally: {traj}")

    script = REPO_ROOT / "get_started" / "replay_multi_scene_render.py"
    spec = importlib.util.spec_from_file_location("rsmsr", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._read_traj_length(traj, "openarm_bimanual_wuji") == 849


@pytest.mark.general
def test_converter_script_round_trips_legacy_input(tmp_path):
    """The converter must rename the robot key and remap finger joints."""
    spec = importlib.util.spec_from_file_location(
        "convert_box_task",
        REPO_ROOT / "scripts" / "convert_box_task_legacy_traj.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    legacy = {
        "openarm_wuji": [
            {
                "init_state": {
                    "openarm_wuji": {
                        "pos": [0, 0, 0],
                        "rot": [1, 0, 0, 0],
                        "dof_pos": {"left_hand_finger1_joint1": 0.1, "openarm_left_joint1": 0.2},
                    },
                    "extra_object": {"pos": [1, 0, 0]},
                },
                "states": [
                    {
                        "openarm_wuji": {"dof_pos": {"right_hand_finger5_joint4": 0.5}},
                        "extra_object": {"pos": [2, 0, 0]},
                    },
                ],
                "actions": [{"dof_pos_target": {"left_hand_finger2_joint3": 0.7}}],
            }
        ]
    }
    src = tmp_path / "legacy.pkl"
    dst = tmp_path / "out.pkl"
    with src.open("wb") as f:
        pickle.dump(legacy, f)

    mod.convert(src, dst)

    with dst.open("rb") as f:
        out = pickle.load(f)
    assert list(out.keys()) == ["openarm_bimanual_wuji"]
    ep = out["openarm_bimanual_wuji"][0]
    assert "openarm_bimanual_wuji" in ep["init_state"]
    assert ep["init_state"]["openarm_bimanual_wuji"]["dof_pos"] == {
        "left_finger1_joint1": 0.1,
        "openarm_left_joint1": 0.2,
    }
    # Non-robot keys preserved.
    assert ep["init_state"]["extra_object"] == {"pos": [1, 0, 0]}
    # States and actions also remapped.
    assert ep["states"][0]["openarm_bimanual_wuji"]["dof_pos"] == {"right_finger5_joint4": 0.5}
    assert ep["actions"][0]["dof_pos_target"] == {"left_finger2_joint3": 0.7}


@pytest.mark.general
def test_task_objects_carry_both_usd_and_mjcf_paths():
    """Each rigid object must expose ``usd_path`` (isaacsim/blender) and
    ``mjcf_path`` (mujoco) so either backend can load the scenario."""
    from metasim.scenario.objects import RigidObjCfg
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    rigid_objs = [o for o in BoxTaskReplayEnv.scenario.objects if isinstance(o, RigidObjCfg)]
    assert len(rigid_objs) == 3
    for obj in rigid_objs:
        assert obj.usd_path and obj.usd_path.endswith(".usd"), f"{obj.name}: usd_path missing"
        assert obj.mjcf_path and obj.mjcf_path.endswith(".xml"), f"{obj.name}: mjcf_path missing"


@pytest.mark.general
def test_replay_cli_advertises_simulator_arg():
    """``--simulator`` must accept isaacsim and mujoco."""
    script = REPO_ROOT / "get_started" / "replay_multi_scene_render.py"
    spec = importlib.util.spec_from_file_location("rsmsr", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser = mod._build_arg_parser()
    action = next(a for a in parser._actions if "--simulator" in a.option_strings)
    assert action.default == "isaacsim"
    assert set(action.choices) == {"isaacsim", "mujoco"}


@pytest.mark.general
def test_prepare_script_stages_all_assets(tmp_path):
    """Running ``prepare_box_task_assets`` against the bundle source
    produces robot MJCF + object MJCFs + canonical traj at the expected
    paths. Uses a synthetic ``--repo-root`` so we don't disturb the
    real ``roboverse_data/`` tree."""
    bundle = Path("/home/ghr/projects/RoboVerse/box_task_replay_render_bundle_clean")
    if not bundle.exists():
        pytest.skip(f"upstream bundle not present at {bundle}")

    spec = importlib.util.spec_from_file_location(
        "prep",
        REPO_ROOT / "scripts" / "prepare_box_task_assets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The script invokes the sibling converter via its committed path —
    # use the real repo root so that import works.
    mod.prepare(bundle.resolve(), REPO_ROOT)

    rdata = REPO_ROOT / "roboverse_data"
    robot_mjcf = rdata / "robots" / "openarm_wuji" / "openarm_wuji.xml"
    assert robot_mjcf.exists()
    text = robot_mjcf.read_text()
    # Mocap bodies stripped (they used to crash mujoco compilation when
    # the robot was attached under another parent).
    assert "mocap_left" not in text and "mocap_right" not in text
    # Joint + actuator names match the cfg convention. Body / mesh names
    # such as ``left_hand_finger1_link1`` may legitimately survive — only
    # the joint and actuator surfaces are part of the runtime contract.
    assert 'joint name="left_hand_finger' not in text
    assert 'position name="left_hand_finger' not in text
    assert 'joint name="left_finger1_joint1"' in text
    # Arm motor names match joint names instead of carrying ``_ctrl`` suffix.
    assert 'motor name="left_joint1_ctrl"' not in text
    assert 'motor name="openarm_left_joint1"' in text

    for obj in ("cardboard_box", "feast_soda_can", "feast_scented_candle"):
        assert (rdata / "assets" / "box_task" / "local_pack_box" / obj / f"{obj}.xml").exists()

    assert (rdata / "trajs" / "box_task" / "task3_openarm_bimanual_wuji_v2.pkl").exists()
