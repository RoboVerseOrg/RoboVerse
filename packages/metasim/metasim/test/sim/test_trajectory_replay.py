"""An episode recorded on MuJoCo, saved to disk and loaded back replays to the recorded states (L0 from a file)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.replay import verify_episode_replay
from metasim.utils.trajectory import check_assets, load_episode, record_episode, save_episode

pytestmark = pytest.mark.mujoco


def _handler(decimation: int = 15):
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler

    scenario = ScenarioCfg(
        robots=[FrankaCfg()],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.05, 0.05, 0.05),
                color=[0.8, 0.1, 0.1],
                default_position=[0.4, 0.0, 0.3],
                physics=PhysicStateType.RIGIDBODY,
            )
        ],
        simulator="mujoco",
        num_envs=1,
        headless=True,
        decimation=decimation,
    )
    h = MujocoHandler(scenario)
    h.launch()
    return h


def test_mujoco_episode_survives_disk_and_replays(tmp_path):
    h = _handler()
    try:
        names = h.get_joint_names("franka", sort=True)
        actions = [torch.zeros(1, len(names)) for _ in range(30)]
        for t, a in enumerate(actions):
            a[0, names.index("panda_joint1")] = 0.02 * t
        episode = record_episode(h, h.get_states(mode="tensor"), actions, seed=3, info={"task": "drive joint 1"})
        assert episode.provenance.simulator == "mujoco" and episode.provenance.seed == 3
        assert episode.provenance.physics_dt == pytest.approx(0.001) and episode.provenance.env_step_s == pytest.approx(
            0.015
        )
        assert episode.provenance.backend_versions.get("mujoco")
        assert "franka" in episode.provenance.assets and check_assets(episode) == {
            k: "ok" for k in check_assets(episode)
        }
        assert episode.joint_names["franka"] == names
        path = save_episode(episode, tmp_path / "episode.npz")
        back = load_episode(path)
        assert back.states[-1].robots["franka"].joint_pos.dtype == torch.float64
        report = verify_episode_replay(h, back, tol=1e-4)
        assert report.passed, str(report)
        assert len(report.per_step) == 31  # the initial state and one entry per action
    finally:
        h.close()


def test_mujoco_replay_refuses_a_different_time_base(tmp_path):
    h = _handler(decimation=15)
    try:
        episode = record_episode(h, h.get_states(mode="tensor"), [torch.zeros(1, 9)] * 3)
        path = save_episode(episode, tmp_path / "ep.npz")
    finally:
        h.close()
    other = _handler(decimation=5)
    try:
        with pytest.raises(ValueError, match="time base differs"):
            verify_episode_replay(other, load_episode(path))
    finally:
        other.close()
