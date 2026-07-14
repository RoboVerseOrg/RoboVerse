"""Unit tests for the fusion pipeline orchestrator command construction.

Pure (no execution): verifies the right stages map to the right existing entry
points with consistent paths threaded between them.
"""

from __future__ import annotations

import pytest

# The orchestrator parses its CLI with tyro at module scope. This has to be an
# importorskip rather than a `requires_optional` marker: the import below happens at
# collection time, before any marker would get a chance to skip the test.
pytest.importorskip(
    "tyro",
    reason='optional dependency not installed: tyro — install with: python -m pip install -e ".[examples]"',
)

from roboverse_learn.fusion.pipeline import Args, build_commands


def _joined(cmds):
    return [" ".join(c) for c in cmds]


def test_default_three_stage_pipeline():
    cmds = build_commands(Args(task="mjlab.lift_cube_yam_v2", robot="yam", name="lift", num_demos=50))
    joined = _joined(cmds)
    assert len(cmds) == 3
    assert "roboverse_learn.fusion.collect" in joined[0]
    assert "data2zarr_dp.py" in joined[1]
    assert "il/train.py" in joined[2]
    # collect reads the default policy.pt; to-zarr reads collect's demo dir;
    # il-train reads to-zarr's output zarr — paths must be threaded consistently.
    assert "outputs/lift/policy.pt" in joined[0]
    assert "roboverse_demo/lift/success" in joined[1]
    assert "data_policy/lift_50.zarr" in joined[2]


def test_rl_train_stage_added():
    cmds = build_commands(
        Args(task="t", name="r", stages=["rl-train", "collect", "to-zarr", "il-train"], rl_iterations=1500)
    )
    joined = _joined(cmds)
    assert len(cmds) == 4
    assert "rsl_rl/ppo.py" in joined[0]
    assert "--max_iterations 1500" in joined[0]


def test_explicit_checkpoint_used_in_collect():
    cmds = build_commands(Args(task="t", name="r", checkpoint="/ckpts/model_499.pt", stages=["collect"]))
    assert "--checkpoint /ckpts/model_499.pt" in " ".join(cmds[0])


def test_keep_all_propagates():
    cmds = build_commands(Args(task="t", name="r", stages=["collect"], keep_all=True))
    assert "--keep_all" in " ".join(cmds[0])
    cmds2 = build_commands(Args(task="t", name="r", stages=["collect"], keep_all=False))
    assert "--keep_all" not in " ".join(cmds2[0])


def test_unknown_stage_fails_fast():
    with pytest.raises(ValueError, match="unknown stage"):
        build_commands(Args(task="t", name="r", stages=["collect", "bogus"]))
