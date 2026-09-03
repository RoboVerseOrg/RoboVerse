"""Backend version policy (``metasim.sim._versions``): the gate refuses unsupported releases, warns
once on untested ones, and ``python -m metasim doctor`` reports every backend."""

from __future__ import annotations

import subprocess
import sys

import pytest

from metasim.constants import SimType
from metasim.sim import _versions as v


def _fake_installed(monkeypatch, table: dict[str, str | None]):
    monkeypatch.setattr(v, "_installed", lambda dist: table.get(dist))


@pytest.mark.general
def test_unsupported_version_raises_with_both_versions_in_the_message(monkeypatch):
    _fake_installed(monkeypatch, {"mujoco": "2.3.7", "dm-control": "1.0.45"})
    monkeypatch.delenv("METASIM_SKIP_VERSION_CHECK", raising=False)
    with pytest.raises(v.BackendVersionError, match=r"mujoco 2\.3\.7 \(supported: >=3\.2,<3\.14\)"):
        v.enforce_backend_versions(SimType.MUJOCO)


@pytest.mark.general
def test_skip_env_downgrades_the_error_to_a_warning(monkeypatch):
    _fake_installed(monkeypatch, {"mujoco": "2.3.7", "dm-control": "1.0.45"})
    monkeypatch.setenv("METASIM_SKIP_VERSION_CHECK", "1")
    report = v.enforce_backend_versions(SimType.MUJOCO)
    assert report.unsupported and report.unsupported[0].requirement.dist == "mujoco"


@pytest.mark.general
def test_untested_version_inside_the_range_is_reported_not_refused(monkeypatch):
    _fake_installed(monkeypatch, {"mujoco": "3.13.0", "dm-control": "1.0.45"})
    monkeypatch.delenv("METASIM_SKIP_VERSION_CHECK", raising=False)
    v._WARNED.clear()
    report = v.enforce_backend_versions(SimType.MUJOCO)
    assert not report.unsupported and [s.requirement.dist for s in report.untested] == ["mujoco"]
    assert report.statuses[0].label == "untested"


@pytest.mark.general
def test_missing_backend_is_not_an_error(monkeypatch):
    _fake_installed(monkeypatch, {})
    report = v.enforce_backend_versions(SimType.SAPIEN3)
    assert not report.installed and report.statuses[0].label == "missing"


@pytest.mark.general
def test_prereleases_count_as_inside_the_range(monkeypatch):
    _fake_installed(
        monkeypatch, {"newton": "1.6.0.dev0", "warp-lang": "1.17.0", "mujoco-warp": "3.12.0", "mujoco": "3.12.0"}
    )
    assert not v.check_backend(SimType.NEWTON).unsupported


@pytest.mark.general
def test_doctor_cli_runs_and_reports_every_backend():
    out = subprocess.run(
        [sys.executable, "-m", "metasim", "doctor", "--json"], capture_output=True, text=True, check=False
    )
    assert out.returncode in (0, 1), out.stderr[-500:]
    import json

    names = {row["backend"] for row in json.loads(out.stdout)}
    assert {"mujoco", "newton", "superdex", "sapien3"} <= names
