"""``metasim.utils.trajectory``: an episode file is self-describing, lossless, validated, and detected by content."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.types import ObjectState, RobotState, TensorState
from metasim.utils.trajectory import (
    FORMAT,
    FORMAT_VERSION,
    EpisodeRecord,
    Provenance,
    check_assets,
    episode_from_states,
    is_episode_file,
    load_episode,
    read_header,
    save_episode,
)

pytestmark = pytest.mark.general


class _Handler:
    """Enough of a handler for ``episode_from_states`` / ``provenance_from_handler``."""

    num_envs = 2
    device = torch.device("cpu")
    physics_dt = 0.001

    @property
    def env_step_s(self):  # the backend contract: physics_dt x decimation
        return self.physics_dt * self.scenario.decimation

    def __init__(self, asset_path: str | None = None):
        robot = SimpleNamespace(name="arm", mjcf_path=asset_path, urdf_path=None)
        self.scenario = SimpleNamespace(
            simulator="mujoco",
            robots=[robot],
            objects=[SimpleNamespace(name="cube", usd_path=None)],
            cameras=[SimpleNamespace(name="cam", width=64, height=48, pos=(1.0, 0.0, 1.0))],
            decimation=3,
            sim_params=SimpleNamespace(dt=None),
        )

    def get_joint_names(self, name, sort=True):
        return ["j_a", "j_b"] if name == "arm" else []

    def get_body_names(self, name, sort=True):
        return ["base", "tip"] if name == "arm" else ["cube"]


def _state(t: float) -> TensorState:
    root = torch.tensor([[t, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    return TensorState(
        robots={
            "arm": RobotState(
                root_state=root.clone(),
                body_names=["base", "tip"],
                body_state=torch.zeros(2, 2, 13, dtype=torch.float64),
                joint_pos=torch.tensor([[t, -t], [t + 1, -t - 1]], dtype=torch.float64),
                joint_vel=torch.zeros(2, 2, dtype=torch.float64),
                joint_pos_target=torch.full((2, 2), 0.25, dtype=torch.float64),
            )
        },
        objects={"cube": ObjectState(root_state=root.clone() + 1.0 * torch.tensor([1.0] + [0.0] * 12))},
        cameras={},
        extras={},
    )


def _record(handler=None) -> EpisodeRecord:
    handler = handler or _Handler()
    states = [_state(0.0), _state(0.1), _state(0.2)]
    actions = [torch.tensor([[0.3, 0.4], [0.5, 0.6]]), torch.tensor([[0.7, 0.8], [0.9, 1.0]])]
    return episode_from_states(handler, states, actions, seed=7, info={"task": "unit"})


def test_round_trip_is_lossless_and_keeps_names_and_provenance(tmp_path):
    rec = _record()
    path = save_episode(rec, tmp_path / "ep")
    assert path.suffix == ".npz"
    back = load_episode(path)
    assert len(back) == 2 and len(back.states) == 3
    for a, b in zip(rec.states, back.states, strict=True):
        assert torch.equal(a.robots["arm"].joint_pos, b.robots["arm"].joint_pos)
        assert torch.equal(a.robots["arm"].root_state, b.robots["arm"].root_state)
        assert torch.equal(a.objects["cube"].root_state, b.objects["cube"].root_state)
        assert b.robots["arm"].joint_pos.dtype == torch.float64
        assert torch.equal(a.robots["arm"].joint_pos_target, b.robots["arm"].joint_pos_target)
        assert b.robots["arm"].body_names == ["base", "tip"]
    assert torch.equal(back.actions[1], torch.tensor([[0.7, 0.8], [0.9, 1.0]]).double())  # what was recorded, exactly
    assert back.joint_names == {"arm": ["j_a", "j_b"]}
    assert back.body_names == {"arm": ["base", "tip"], "cube": ["cube"]}
    assert back.entities == {"robots": ["arm"], "objects": ["cube"]}
    p = back.provenance
    assert (p.simulator, p.num_envs, p.decimation, p.seed) == ("mujoco", 2, 3, 7)
    assert p.physics_dt == 0.001 and p.env_step_s == pytest.approx(0.003)
    assert p.metasim_version and p.python and p.torch and p.numpy and p.platform and p.created_at
    assert back.cameras["cam"]["width"] == 64 and back.cameras["cam"]["pos"] == [1.0, 0.0, 1.0]
    assert back.info == {"task": "unit"}


def test_header_states_the_conventions_and_is_readable_without_arrays(tmp_path):
    path = save_episode(_record(), tmp_path / "ep.npz")
    header = read_header(path)
    assert header["format"] == FORMAT and header["format_version"] == FORMAT_VERSION
    assert header["quaternion"] == "wxyz"
    assert header["root_state_layout"] == ["pos_xyz", "quat_wxyz", "lin_vel_world", "ang_vel_world"]
    assert {"robots/arm/joint_vel", "actions"} <= set(header["arrays"])
    with np.load(path, allow_pickle=False) as data:  # no pickle anywhere in the file
        assert set(data.files) >= {"header", "actions", "robots/arm/root_state", "objects/cube/root_state"}


def test_detection_is_by_content_not_name(tmp_path):
    path = save_episode(_record(), tmp_path / "franka_v2.npz")  # a misleading legacy-looking name
    assert is_episode_file(path)
    other = tmp_path / "not_an_episode.npz"
    np.savez(other, x=np.zeros(3))
    assert not is_episode_file(other)
    with pytest.raises(ValueError, match=r"not a roboverse\.episode file"):
        read_header(other)


def test_newer_format_version_is_refused(tmp_path):
    path = save_episode(_record(), tmp_path / "ep.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    header = json.loads(str(arrays["header"]))
    header["format_version"] = FORMAT_VERSION + 1
    arrays["header"] = np.array(json.dumps(header))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="newer than this reader"):
        load_episode(path)


def test_validate_rejects_inconsistent_records():
    rec = _record()
    rec.actions.append(torch.zeros(2, 2))  # T + 1 states for T + 1 actions
    with pytest.raises(ValueError, match=r"expected T \+ 1"):
        rec.validate()
    rec = _record()
    rec.states[1].robots["arm"].root_state[:, 3:7] = 0.0
    with pytest.raises(ValueError, match="not unit length"):
        rec.validate()
    rec = _record()
    rec.actions[0] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="expected num_envs=2"):
        rec.validate()


def test_assets_are_hashed_and_checked(tmp_path, monkeypatch):
    asset = tmp_path / "arm.xml"
    asset.write_text("<mujoco/>")
    rec = _record(_Handler(asset_path=str(asset)))
    entry = rec.provenance.assets["arm"]["mjcf_path"]
    assert entry["path"] == entry["resolved"] == str(asset) and len(entry["sha256"]) == 64 and entry["bytes"] == 9
    assert check_assets(rec) == {"arm.mjcf_path": "ok"}
    asset.write_text("<mujoco><option timestep='0.002'/></mujoco>")
    assert check_assets(rec) == {"arm.mjcf_path": "changed"}
    asset.unlink()
    assert check_assets(rec) == {"arm.mjcf_path": "missing"}


def test_provenance_is_a_plain_dataclass_with_jsonable_fields():
    p = _record().provenance
    assert isinstance(p, Provenance)
    json.dumps(p.__dict__)  # every field survives a JSON header


def test_replay_refuses_a_handler_with_a_different_num_envs_or_unknown_time_base():
    """A broadcast replay of a 1-env record on an N-env handler would compare nothing (``state_distance``
    skips shape mismatches) and pass vacuously; an unknown time base cannot be validated at all."""
    from metasim.utils.replay import verify_episode_replay

    rec = _record()  # 2 envs, 3 ms env step

    class _WrongEnvs(_Handler):
        num_envs = 3

    with pytest.raises(ValueError, match="episode has 2 env\\(s\\), handler has 3"):
        verify_episode_replay(_WrongEnvs(), rec)

    class _NoTimeBase(_Handler):
        physics_dt = None

        @property
        def env_step_s(self):
            return None

    with pytest.raises(ValueError, match="time base unknown"):
        verify_episode_replay(_NoTimeBase(), rec)

    class _OtherStep(_Handler):
        physics_dt = 0.002  # 6 ms env step vs the recorded 3 ms

    with pytest.raises(ValueError, match="time base differs"):
        verify_episode_replay(_OtherStep(), rec)


def test_replay_refuses_entities_or_joints_the_handler_does_not_have_and_never_passes_vacuously():
    """A record naming an entity or a joint layout the handler lacks would be compared on nothing
    (``state_distance`` skips absent / shape-mismatched keys); that is refused up front, and a replay
    that compared zero quantities is reported as failed, not passed."""
    from metasim.utils.replay import ReplayReport, Trajectory, verify_action_replay, verify_episode_replay

    rec = _record()

    class _NoCube(_Handler):
        def __init__(self):
            super().__init__()
            self.scenario.objects = []

    with pytest.raises(ValueError, match=r"episode objects \['cube'\] are not in the handler's scenario"):
        verify_episode_replay(_NoCube(), rec)

    class _OtherJoints(_Handler):
        def get_joint_names(self, name, sort=True):
            return ["j_a", "j_b", "j_c"] if name == "arm" else []

    with pytest.raises(ValueError, match="joint names of 'arm' differ"):
        verify_episode_replay(_OtherJoints(), rec)

    class _Empty:
        """A handler whose states share nothing with the record."""

        def set_states(self, s):
            pass

        def set_dof_targets(self, a):
            pass

        def simulate(self):
            pass

        def get_states(self, mode="tensor"):
            return TensorState(objects={}, robots={}, cameras={}, extras={})

    report = verify_action_replay(_Empty(), Trajectory(states=rec.states, actions=rec.actions))
    assert isinstance(report, ReplayReport) and report.compared_keys == 0 and report.passed is False
    assert "nothing compared" in report.worst_key


def test_asset_files_distinguish_primitives_from_missing_files(tmp_path):
    """A primitive shape configures no asset (nothing to hash, nothing to warn about); a config whose
    file is absent on this machine is reported as unresolved so the recorder can say so."""
    from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
    from metasim.utils.trajectory import _asset_files

    cube = PrimitiveCubeCfg(name="cube", size=(0.1, 0.1, 0.1), color=(1.0, 0.0, 0.0))
    assert _asset_files(cube, "mujoco") == ({}, [])

    real = tmp_path / "box.xml"
    real.write_text("<mujoco/>")
    present = RigidObjCfg(name="box", mjcf_path=str(real))
    files, unresolved = _asset_files(present, "mujoco")
    assert unresolved == [] and files["asset"] == (str(real), real)

    absent = RigidObjCfg(name="ghost", mjcf_path=str(tmp_path / "missing.xml"))
    files, unresolved = _asset_files(absent, "mujoco")
    assert files == {} and unresolved == [str(tmp_path / "missing.xml")]


def test_assets_configured_relative_to_home_are_checked_on_another_home(tmp_path, monkeypatch):
    """The record keeps the *configured* path and re-resolves it where it is replayed: a ``~``-relative
    asset recorded under one home is found under another home that holds the same file."""
    import shutil

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "arm.xml").write_text("<mujoco/>")
    monkeypatch.setenv("HOME", str(home_a))
    rec = _record(_Handler(asset_path="~/arm.xml"))
    entry = rec.provenance.assets["arm"]["mjcf_path"]
    assert entry["path"] == "~/arm.xml" and entry["resolved"] == str(home_a / "arm.xml")
    shutil.copy(home_a / "arm.xml", home_b / "arm.xml")
    monkeypatch.setenv("HOME", str(home_b))
    assert check_assets(rec) == {"arm.mjcf_path": "ok"}


def test_backend_failures_while_reporting_the_time_base_are_not_recorded_as_unknown():
    """Only a backend that has not resolved its step answers None; a failure propagates."""
    from metasim.utils.trajectory import env_step_seconds

    class _Broken:
        @property
        def env_step_s(self):
            raise RuntimeError("worker 0 died")

    class _Unresolved:
        pass

    with pytest.raises(RuntimeError, match="worker 0 died"):
        env_step_seconds(_Broken())
    assert env_step_seconds(_Unresolved()) is None


def test_ee_state_never_returns_a_misaligned_trajectory():
    """A recording without body states yields no EE states (the writer can say so); one where only some
    frames carry them is refused, because skipping those frames shifted every later EE state."""
    from types import SimpleNamespace

    from metasim.utils.kinematics import get_ee_state_from_list

    robot = SimpleNamespace(
        name="arm", ee_body_name="hand", ee_joint_names=["f1"], gripper_open_q=[0.04], gripper_close_q=[0.0]
    )
    with_body = {
        "robots": {
            "arm": {
                "body": {"hand": {"pos": [0.0, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]}},
                "dof_pos": {"f1": 0.04},
            }
        }
    }
    without = {"robots": {"arm": {"dof_pos": {"f1": 0.04}}}}
    assert get_ee_state_from_list([without, without], robot, tensorize=True).shape == (0, 7)
    assert get_ee_state_from_list([], robot, tensorize=True).shape == (0, 7) and get_ee_state_from_list([], robot) == []
    assert get_ee_state_from_list([without], robot, tensorize=True, use_rpy=False).shape == (0, 8)
    assert get_ee_state_from_list([with_body, with_body], robot, tensorize=True).shape == (2, 7)
    with pytest.raises(ValueError, match=r"1 of 3 frames carry no body states .*first at step 1"):
        get_ee_state_from_list([with_body, without, with_body], robot, tensorize=True)
