"""End-to-end RL->IL data-contract test (no simulator).

Synthesises the per-step ``DictEnvState`` list that
``roboverse_learn.fusion.rl_to_demo`` produces during a rollout, writes it with
the canonical :func:`metasim.utils.save_util.save_demo`, then runs the *real*
``roboverse_learn/il/data2zarr_dp.py`` and loads the resulting zarr — proving a
collected demo is consumable by the existing IL pipeline unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr")


def _fake_robot_cfg():
    """Minimal stand-in with the fields get_ee_state_from_list reads."""
    return types.SimpleNamespace(
        name="r",
        ee_body_name="ee",
        ee_joint_names=["f1"],
        gripper_open_q=[0.04],
        gripper_close_q=[0.0],
    )


def _synthetic_demo(num_steps: int = 6, h: int = 32, w: int = 32):
    """A list[DictEnvState] shaped exactly like state_tensor_to_nested output."""
    demo = []
    for t in range(num_steps):
        q = float(t) * 0.01
        robot = {
            "dof_pos": {"j0": q, "j1": q + 0.1, "f1": 0.04},
            "dof_pos_target": {"j0": q + 0.005, "j1": q + 0.105, "f1": 0.04},
            "pos": torch.tensor([0.0, 0.0, 0.1]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "vel": torch.zeros(3),
            "ang_vel": torch.zeros(3),
            "body": {"ee": {"pos": torch.tensor([0.1, 0.0, 0.2]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])}},
        }
        cam = {"rgb": (torch.ones(h, w, 3) * (t * 10 % 255)).to(torch.uint8)}
        demo.append({"robots": {"r": robot}, "cameras": {"cam": cam}})
    return demo


def test_collected_demo_is_il_consumable(tmp_path):
    from metasim.utils.save_util import save_demo

    # 1) Two synthetic episodes written exactly as the collector would.
    success_dir = tmp_path / "success"
    n_eps, steps = 2, 6
    for i in range(n_eps):
        save_demo(str(success_dir / f"demo_{i:04d}"), _synthetic_demo(steps), _fake_robot_cfg(), "fusion-contract-test")
        assert (success_dir / f"demo_{i:04d}" / "metadata.json").exists()
        assert (success_dir / f"demo_{i:04d}" / "rgb.mp4").exists()

    # metadata.json carries the keys data2zarr reads.
    meta = json.loads((success_dir / "demo_0000" / "metadata.json").read_text())
    assert len(meta["joint_qpos"]) == steps
    assert len(meta["joint_qpos"][0]) == 3  # j0, j1, f1 sorted
    assert len(meta["joint_qpos_target"]) == steps

    # 2) Run the REAL data2zarr conversion (cwd=tmp so data_policy/ lands here).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo, "roboverse_learn", "il", "data2zarr_dp.py")
    res = subprocess.run(
        [
            sys.executable,
            script,
            "--task_name",
            "fusion_contract",
            "--metadata_dir",
            str(success_dir),
            "--expert_data_num",
            str(n_eps),
            "--observation_space",
            "joint_pos",
            "--action_space",
            "joint_pos",
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"data2zarr failed:\n{res.stdout}\n{res.stderr}"

    # 3) Load the zarr and assert the IL dataset contract.
    zpath = tmp_path / "data_policy" / f"fusion_contract_{n_eps}.zarr"
    assert zpath.exists(), f"zarr not written at {zpath}"
    root = zarr.open(str(zpath), mode="r")
    state = root["data"]["state"][:]
    action = root["data"]["action"][:]
    head = root["data"]["head_camera"][:]
    ends = root["meta"]["episode_ends"][:]

    assert state.shape == (n_eps * steps, 3), state.shape
    assert action.shape == (n_eps * steps, 3), action.shape
    assert head.shape[0] == n_eps * steps and head.shape[1] == 3, head.shape  # NCHW
    assert ends.tolist() == [steps, 2 * steps], ends.tolist()
    # joint_qpos is sorted by joint name -> ["f1", "j0", "j1"], so the monotonic
    # j0 we wrote lands in column 1 (column 0 is the constant finger f1=0.04).
    assert np.allclose(state[:steps, 0], 0.04)
    assert np.all(np.diff(state[:steps, 1]) > 0)
