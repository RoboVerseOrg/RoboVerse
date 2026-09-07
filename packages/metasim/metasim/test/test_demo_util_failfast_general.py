"""``get_traj`` must fail fast with a documented exception, not silently.

Two regressions, both of which previously raised an exception OUTSIDE the
``(FileNotFoundError, KeyError, ValueError)`` set that task base classes catch
(e.g. ``LiberoBaseTask._get_initial_states``), so they surfaced as opaque
crashes far from the real cause instead of a graceful fallback:

- v1-format trajectory: the v1 branch only logged a warning and fell off the
  end, returning ``None`` → callers ``init_states, _, _ = get_traj(...)`` hit
  ``TypeError: cannot unpack non-iterable NoneType``.
- empty trajectory: ``data[0]`` on an empty list raised ``IndexError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.utils.demo_util import get_traj

pytestmark = pytest.mark.general
from metasim.utils.demo_util.demo_util import detect_traj_format


def test_get_traj_missing_file_raises_filenotfound(tmp_path):
    """An absent path is reported as missing (FileNotFoundError), not as an unsupported format."""
    from types import SimpleNamespace

    robot = SimpleNamespace(name="franka")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_traj(str(tmp_path / "demos" / "franka_v1.pkl.gz"), robot)


def test_get_traj_v1_format_raises_valueerror_not_silent_none(tmp_path):
    """A file whose content is neither an episode nor the legacy dict-of-lists layout raises ValueError
    naming the accepted formats; the old branch logged a warning and returned None (callers then hit
    ``TypeError: cannot unpack non-iterable NoneType``)."""
    import pickle
    from types import SimpleNamespace

    v1 = tmp_path / "franka_v1.pkl"
    with open(v1, "wb") as f:
        pickle.dump([{"obs": 1, "action": 2}], f)  # a v1-style flat list
    with pytest.raises(ValueError, match="Unsupported trajectory format"):
        get_traj(str(v1), SimpleNamespace(name="franka"))


def test_get_traj_v2_empty_trajectory_raises_valueerror(monkeypatch, tmp_path):
    """An empty trajectory list must raise ValueError before indexing data[0]
    (previously an IndexError that escaped the caller's catch tuple)."""
    import metasim.utils.demo_util.demo_util_v2 as v2

    robot = SimpleNamespace(name="franka")
    f = tmp_path / "franka_v2.pkl"  # must exist + endswith _v2.pkl for the asserts
    f.write_bytes(b"x")
    monkeypatch.setattr(v2, "load_traj_file", lambda p: {"franka": []})
    with pytest.raises(ValueError, match="empty"):
        v2.get_traj_v2(str(f), robot)


def _v2_file(path, robot_name="franka"):
    import pickle

    demo = {
        "init_state": {robot_name: {"pos": [0, 0, 0], "rot": [1, 0, 0, 0], "dof_pos": {"j": 0.0}}},
        "actions": [{"dof_pos_target": {"j": 0.1}}],
        "states": None,
    }
    with open(path, "wb") as f:
        pickle.dump({robot_name: [demo]}, f)
    return path


def test_v2_format_is_recognised_by_content_not_by_the_path(tmp_path):
    """A v2 file under a path without the substring 'v2' loads; the substring used to decide the parser."""
    from types import SimpleNamespace

    robot = SimpleNamespace(name="franka", joint_limits={"j": (-1, 1)})
    path = _v2_file(tmp_path / "relocated" / "franka.pkl") if (tmp_path / "relocated").mkdir() is None else None
    assert detect_traj_format(str(path))[0] == "v2"
    init_states, actions, _states = get_traj(str(path), robot, v2_as_v3=False)
    assert len(init_states) == 1 and actions[0][0]["dof_pos_target"] == {"j": 0.1}
    # a directory is searched for the robot's file, with or without the legacy suffix
    assert detect_traj_format(str(tmp_path / "relocated"), "franka")[:2] == ("v2", str(path))
    assert get_traj(str(tmp_path / "relocated"), robot, v2_as_v3=False)[1][0][0]["dof_pos_target"] == {"j": 0.1}


def test_a_path_with_v2_in_it_but_the_wrong_content_is_rejected(tmp_path):
    import pickle
    from types import SimpleNamespace

    bad = tmp_path / "exp_v2" / "franka_v2.pkl"
    bad.parent.mkdir()
    with open(bad, "wb") as f:
        pickle.dump({"not": "a trajectory"}, f)
    assert detect_traj_format(str(bad))[0] == "unknown"
    with pytest.raises(ValueError, match="Unsupported trajectory format"):
        get_traj(str(bad), SimpleNamespace(name="franka"))


def test_an_episode_file_is_pointed_at_the_right_loader(tmp_path):
    from types import SimpleNamespace

    import torch

    from metasim.types import RobotState, TensorState
    from metasim.utils.trajectory import episode_from_states, save_episode

    handler = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scenario=SimpleNamespace(
            simulator="mujoco",
            robots=[SimpleNamespace(name="r")],
            objects=[],
            cameras=[],
            decimation=1,
            sim_params=SimpleNamespace(dt=None),
        ),
        get_joint_names=lambda name, sort=True: ["j"],
        get_body_names=lambda name, sort=True: [],
    )
    root = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6], dtype=torch.float64)
    state = TensorState(
        robots={"r": RobotState(root_state=root, joint_pos=torch.zeros(1, 1, dtype=torch.float64))},
        objects={},
        cameras={},
        extras={},
    )
    from metasim.utils.trajectory import EpisodeFileError

    path = save_episode(episode_from_states(handler, [state], [], seed=None), tmp_path / "franka_v2.npz")
    with pytest.raises(EpisodeFileError, match="load_episode"):
        get_traj(str(path), SimpleNamespace(name="r"))
    assert not issubclass(EpisodeFileError, ValueError)  # the task bases swallow ValueError as "no demo"


def test_legacy_layout_diagnostics_reach_the_user(tmp_path):
    """An empty demo list, a missing init_state or an absent robot entry are legacy-layout problems and
    are reported as such (not as an unsupported format)."""
    import pickle
    from types import SimpleNamespace

    robot = SimpleNamespace(name="franka", joint_limits={"j": (-1, 1)})
    empty = tmp_path / "empty.pkl"
    with open(empty, "wb") as f:
        pickle.dump({"franka": []}, f)
    with pytest.raises(ValueError, match="empty"):
        get_traj(str(empty), robot, v2_as_v3=False)
    other = tmp_path / "other.pkl"
    with open(other, "wb") as f:
        pickle.dump({"h1": [{"init_state": {}, "actions": []}]}, f)
    with pytest.raises(KeyError, match="robots in the file: \\['h1'\\]"):
        get_traj(str(other), robot, v2_as_v3=False)


def test_directory_resolution_ignores_sidecars_named_after_the_robot(tmp_path):
    """``franka.json`` (a config) next to ``franka.pkl`` (the trajectory) must not shadow it, and a
    directory holding only an episode file is pointed at the episode loader."""
    import json
    from types import SimpleNamespace

    robot = SimpleNamespace(name="franka", joint_limits={"j": (-1, 1)})
    d = tmp_path / "ds"
    d.mkdir()
    (d / "franka.json").write_text(json.dumps({"name": "franka"}))
    _v2_file(d / "franka.pkl")
    assert get_traj(str(d), robot, v2_as_v3=False)[1][0][0]["dof_pos_target"] == {"j": 0.1}
    fmt, path, _ = detect_traj_format(str(d), "franka")
    assert fmt == "v2" and path.endswith("franka.pkl")


def test_state_nested_to_v2_flattens_one_namespace_keeps_the_v2_keys_and_refuses_a_clash():
    from metasim.utils.demo_util.demo_util_v2 import state_nested_to_v2

    nested = {
        "objects": {"cube": {"pos": [1.0], "vel": [0.0], "body": {"b": {}}}},
        "robots": {
            "arm": {"pos": [2.0], "dof_pos": {"j": 0.1}, "dof_pos_target": {"j": 0.2}, "dof_torque": {"j": 3.0}}
        },
    }
    assert state_nested_to_v2(nested) == {"cube": {"pos": [1.0]}, "arm": {"pos": [2.0], "dof_pos": {"j": 0.1}}}
    with pytest.raises(ValueError, match=r"both: \['arm'\]"):
        state_nested_to_v2({"objects": {"arm": {}}, "robots": {"arm": {}}})


def _legacy_demo(steps: int = 3) -> list[dict]:
    """Two robots and two cameras, shaped like ``state_tensor_to_nested`` output; robot ``r`` reports targets."""
    import torch

    demo = []
    for t in range(steps):
        q = t * 0.01
        r = {
            "dof_pos": {"j0": q, "j1": q + 0.1, "f1": 0.04},
            "dof_pos_target": {"j0": q + 0.005, "j1": q + 0.105, "f1": 0.04},
            "pos": torch.zeros(3),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "vel": torch.zeros(3),
            "ang_vel": torch.zeros(3),
            "body": {"ee": {"pos": torch.zeros(3), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])}},
        }
        other = {"dof_pos": {"j0": 9.0, "j1": 9.0, "f1": 9.0}}  # listed first, no targets
        cam = {"rgb": torch.full((8, 8, 3), t * 10, dtype=torch.uint8)}
        cam2 = {"rgb": torch.full((8, 8, 3), 200, dtype=torch.uint8)}  # distinct pixels: the choice is visible
        demo.append({"robots": {"other": other, "r": r}, "cameras": {"cam": cam, "cam2": cam2}})
    return demo


def _robot_cfg(name="r"):
    return SimpleNamespace(
        name=name, ee_body_name="ee", ee_joint_names=["f1"], gripper_open_q=[0.04], gripper_close_q=[0.0]
    )


def test_save_demo_writes_the_configured_robot_and_names_what_it_leaves_out(tmp_path):
    """The legacy format holds one robot: the configured one (the ``ee_state`` column already followed it), in
    every column including the targets, never whichever robot the dict lists first; the robots and cameras
    left out are named once per process; a missing robot or camera refuses."""
    import json

    import imageio.v2 as iio
    from loguru import logger

    from metasim.utils.log import reset_warn_once
    from metasim.utils.save_util import save_demo

    reset_warn_once()
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        save_demo(str(tmp_path / "d"), _legacy_demo(), _robot_cfg(), "t")
        save_demo(str(tmp_path / "d2"), _legacy_demo(), _robot_cfg(), "t")
        save_demo(str(tmp_path / "h"), _legacy_demo(2), _robot_cfg(), "t", camera_name="cam2")  # an explicit choice
    finally:
        logger.remove(sink)
    assert any("['cam', 'other']" in w for w in warnings), "the explicit choice leaves the first camera out"
    frame = iio.mimread(str(tmp_path / "h" / "rgb.mp4"))[0]
    assert abs(float(frame.mean()) - 200.0) < 8.0, "rgb.mp4 holds cam2's pixels (lossy video, hence the slack)"
    meta = json.loads((tmp_path / "d" / "metadata.json").read_text())
    assert meta["joint_qpos"][0] == pytest.approx([0.04, 0.0, 0.1]), "robot 'r' (the config), not 'other' (first)"
    assert meta["joint_qpos_target"][0] == pytest.approx([0.04, 0.015, 0.115]), "the next frame's target, of 'r'"
    left_out = [w for w in warnings if "['cam2', 'other']" in w]
    assert len(left_out) == 1, "named once per process, not once per demo"

    with pytest.raises(ValueError, match=r"'absent' .* is not in the demo"):
        save_demo(str(tmp_path / "e"), _legacy_demo(2), _robot_cfg("absent"), "t")
    with pytest.raises(ValueError, match=r"camera 'nope' is not in the demo"):
        save_demo(str(tmp_path / "f"), _legacy_demo(2), _robot_cfg(), "t", camera_name="nope")
    demo = _legacy_demo(2)
    for step in demo:
        step["cameras"] = {}
    with pytest.raises(ValueError, match="needs a camera"):
        save_demo(str(tmp_path / "g"), demo, _robot_cfg(), "t")
    # the target column is the next frame's target, so a post-reset first frame without one is fine
    demo = _legacy_demo(3)
    demo[0]["robots"]["r"]["dof_pos_target"] = None
    save_demo(str(tmp_path / "i"), demo, _robot_cfg(), "t")
    meta = json.loads((tmp_path / "i" / "metadata.json").read_text())
    assert meta["joint_qpos_target"][0] == pytest.approx([0.04, 0.015, 0.115])
    demo[1]["robots"]["r"]["dof_pos_target"] = None  # frame 1's target is read for frame 0: a partial column
    with pytest.raises(ValueError, match=r"no dof_pos_target on frames \[0\]"):
        save_demo(str(tmp_path / "j"), demo, _robot_cfg(), "t")
    for frame in demo:
        frame["robots"]["r"]["dof_pos_target"] = None
    save_demo(str(tmp_path / "k"), demo, _robot_cfg(), "t")  # no targets at all: an empty column
    assert json.loads((tmp_path / "k" / "metadata.json").read_text())["joint_qpos_target"] == [None] * 3
