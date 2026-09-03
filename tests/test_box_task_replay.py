"""Integration tests for the box_task replay task and replay CLI."""

from __future__ import annotations

import importlib.util
import os
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
def test_task_uses_short_robot_name():
    """The scenario must reference the short ``openarm_wuji`` robot name —
    that's what the upstream trajectory pkl keys by, and the cfg ships a
    matching ``OpenarmWujiCfg`` alias for it."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    robot = BoxTaskReplayEnv.scenario.robots[0]
    assert robot.name == "openarm_wuji"


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
def test_traj_uses_bundle_naming_conventions():
    """The pkl is the bundle's v2 recording, used directly. Top-level key
    matches the robot's short ``name``, and finger joints retain the
    legacy ``*_hand_finger*`` keys (which the cfg also uses)."""
    from roboverse_pack.tasks.box_task.box_task_replay import BoxTaskReplayEnv

    traj = REPO_ROOT / BoxTaskReplayEnv.traj_filepath
    if not traj.exists():
        pytest.skip(f"trajectory not present locally: {traj}")

    with traj.open("rb") as f:
        data = pickle.load(f)

    assert list(data.keys()) == ["openarm_wuji"]
    ep = data["openarm_wuji"][0]
    assert {"init_state", "states", "actions"}.issubset(ep.keys())
    assert len(ep["states"]) == 849

    sample_state = ep["states"][0]["openarm_wuji"]
    sample_dof = sample_state["dof_pos"]
    expected = {
        f"{side}_hand_finger{i}_joint{j}" for side in ("left", "right") for i in range(1, 6) for j in range(1, 5)
    }
    assert expected.issubset(sample_dof.keys())


@pytest.mark.general
def test_traj_loads_via_get_traj_without_runtime_remap():
    """``get_traj`` consumes the bundle pkl directly — no per-frame
    fixup hook on the task class."""
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
    assert "openarm_wuji" in frame0["robots"]

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

    assert mod._read_traj_length(traj, "openarm_wuji") == 849


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
    produces robot MJCF + object MJCFs + the staged traj at the expected
    paths. Uses a synthetic ``--repo-root`` so we don't disturb the
    real ``roboverse_data/`` tree."""
    bundle_env = os.environ.get("BOX_TASK_REPLAY_BUNDLE")
    if not bundle_env:
        pytest.skip("set $BOX_TASK_REPLAY_BUNDLE to the upstream box_task_replay_render_bundle_clean dir")
    bundle = Path(bundle_env)
    if not bundle.exists():
        pytest.skip(f"upstream bundle not present at {bundle}")

    spec = importlib.util.spec_from_file_location(
        "prep",
        REPO_ROOT / "scripts" / "prepare_box_task_assets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.prepare(bundle.resolve(), tmp_path)

    rdata = tmp_path / "roboverse_data"
    robot_mjcf = rdata / "robots" / "openarm_wuji" / "openarm_wuji.xml"
    assert robot_mjcf.exists()
    text = robot_mjcf.read_text()
    # Mocap bodies stripped (they used to crash mujoco compilation when
    # the robot was attached under another parent).
    assert "mocap_left" not in text and "mocap_right" not in text
    # Finger joint names are preserved verbatim — they already match
    # the robot cfg.
    assert 'joint name="left_hand_finger1_joint1"' in text
    # Arm motor names rewritten to match joint names (cfg requires
    # actuator-name == joint-name).
    assert 'motor name="left_joint1_ctrl"' not in text
    assert 'motor name="openarm_left_joint1"' in text

    for obj in ("cardboard_box", "feast_soda_can", "feast_scented_candle"):
        assert (rdata / "assets" / "box_task" / "local_pack_box" / obj / f"{obj}.xml").exists()

    assert (rdata / "trajs" / "box_task" / "task3_openarm_wuji_v2.pkl").exists()
