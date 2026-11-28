"""The Robots page renders the robot result files (self-check sweep, cross-engine parity) and nothing by hand."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "docs" / "source" / "dataset_benchmark" / "dataset" / "generate_robot_docs.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generate_robot_docs", GEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_robot_docs"] = module
    spec.loader.exec_module(module)
    return module


def _check_file(path: Path, sim: str, ok: bool) -> None:
    rows = [
        {"check": c, "ok": True, "detail": "fine"} for c in ("config", "assets", "kinematics", "limits", "actuators")
    ]
    rows.append({"check": "hold", "ok": ok, "detail": "drift 0.3 rad | base moved" if not ok else "still"})
    path.write_text(
        json.dumps({"simulator": sim, "robots": [{"robot": "toy", "simulator": sim, "ok": ok, "rows": rows}]})
    )


def _parity_file(path: Path, *, ok: bool, judged: bool = True, note: str | None = None) -> None:
    data = {
        "robot": "toy",
        "sim_a": "mujoco",
        "sim_b": "isaacsim",
        "steps": 400,
        "control_dt": 0.015,
        "peak_velocity": 0.63,
        "judged": judged,
        "ok": ok,
        "rows": [
            {
                "joint": "j1",
                "kind": "hinge",
                "max_diff": 0.001,
                "rms_diff": 0.0,
                "tracking_a": 0.1,
                "tracking_b": 0.1,
                "tol": 0.05,
                "lag_a": 6,
                "lag_b": 6,
                "ok": True,
            },
            {
                "joint": "f",
                "kind": "slide",
                "max_diff": 0.02,
                "rms_diff": 0.0,
                "tracking_a": 0.1,
                "tracking_b": 0.1,
                "tol": 0.005,
                "lag_a": 6,
                "lag_b": 6,
                "ok": ok,
            },
        ],
    }
    if note:
        data["note"] = note
    path.write_text(json.dumps(data))


def test_newest_run_per_simulator_and_engine_pair_is_rendered(gen, tmp_path):
    (tmp_path / "2026-11-01-mujoco-check").mkdir()
    (tmp_path / "2026-11-28-mujoco-check").mkdir()
    (tmp_path / "2026-11-28-isaacsim-check").mkdir()
    (tmp_path / "2026-11-28-parity-mujoco-isaacsim").mkdir()
    (tmp_path / "notes").mkdir()
    _check_file(tmp_path / "2026-11-01-mujoco-check" / "robots.json", "mujoco", ok=True)
    _check_file(tmp_path / "2026-11-28-mujoco-check" / "robots.json", "mujoco", ok=False)
    _check_file(tmp_path / "2026-11-28-isaacsim-check" / "robots.json", "isaacsim", ok=True)
    _parity_file(tmp_path / "2026-11-28-parity-mujoco-isaacsim" / "toy.json", ok=False)
    _parity_file(tmp_path / "2026-11-28-parity-mujoco-isaacsim" / "toy__full_range.json", ok=True, judged=False)

    checks = gen.latest_dirs(gen._CHECK_DIR, tmp_path)
    assert {k: v[0] for k, v in checks.items()} == {"mujoco": "2026-11-28", "isaacsim": "2026-11-28"}
    page = gen.render(tmp_path)
    assert "## Configs" in page and "`franka_cfg`" in page, "the config table still comes from the pack"
    assert "### Self-check on mujoco (2026-11-28)" in page and "### Self-check on isaacsim (2026-11-28)" in page
    assert "2026-11-01" not in page, "only the newest sweep per simulator is shown"
    assert "| `toy` | ok | ok | ok | ok | ok | FAIL | FAIL |" in page
    assert "- `toy` / hold: drift 0.3 rad / base moved" in page, "a pipe in a detail cannot break the table"
    assert "### Cross-engine parity: mujoco vs isaacsim (2026-11-28)" in page
    assert "| `toy` | default | 400 | 0.63/s | `f` | 0.0200 m | FAIL |" in page
    assert "| `toy` | full range | 400 | 0.63/s | `f` | 0.0200 m | reported, not judged (floating base) |" in page


def test_committed_page_matches_the_generator(gen):
    """What `--check` enforces in CI, asserted here too so a stale page fails the suite with a readable diff."""
    committed = (GEN.parent / "robots.md").read_text()
    assert committed == gen.render(), "run python docs/source/dataset_benchmark/dataset/generate_robot_docs.py"


def test_no_hand_written_number_outside_the_generated_tables(gen):
    """The hand-written prose of the page is only the header; every table row is generated."""
    page = (GEN.parent / "robots.md").read_text()
    header = page.split("## Configs", 1)[0]
    assert not re.search(r"\|\s*\d", header), "numbers belong in generated tables, not the header"
