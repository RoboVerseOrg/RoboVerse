"""Backend version policy (``metasim.sim._versions``): the gate refuses unsupported releases, warns
once on untested ones, and ``python -m metasim doctor`` reports every backend."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement as PackagingRequirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import metasim
from metasim.constants import SimType
from metasim.sim import _versions as v
from metasim.sim._versions import BACKEND_REQUIREMENTS


def _fake_installed(monkeypatch, table: dict[str, str | None]):
    monkeypatch.setattr(v, "_installed", lambda dist: table.get(dist))


@pytest.mark.general
def test_unsupported_version_raises_with_both_versions_in_the_message(monkeypatch):
    _fake_installed(monkeypatch, {"mujoco": "2.3.7", "dm-control": "1.0.45"})
    monkeypatch.delenv("METASIM_SKIP_VERSION_CHECK", raising=False)
    with pytest.raises(v.BackendVersionError, match=r"mujoco 2\.3\.7 \(supported: >=3\.2,<3\.18\)"):
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
def test_a_backend_that_is_not_installed_is_not_judged_by_its_shared_dependencies(monkeypatch):
    """Newton is absent and the MuJoCo on the box is newer than Newton's range: that is "newton is not
    installed", not a version error about mujoco (CI with mujoco 3.17 and no newton hit this)."""
    _fake_installed(monkeypatch, {"mujoco": "3.17.0", "newton": None, "warp-lang": None, "mujoco-warp": None})
    monkeypatch.delenv("METASIM_SKIP_VERSION_CHECK", raising=False)
    report = v.enforce_backend_versions(SimType.NEWTON)
    assert not report.installed and [s.requirement.dist for s in report.unsupported] == ["mujoco"]


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


def _extra_pins(extra: str) -> dict[str, SpecifierSet]:
    """``{dist: specifier}`` of one ``[project.optional-dependencies]`` extra of ``packages/metasim``."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - CI runs 3.10 and 3.11
        import tomli as tomllib
    pyproject = Path(metasim.__file__).resolve().parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    pins: dict[str, SpecifierSet] = {}
    for line in extras.get(extra, ()):
        req = PackagingRequirement(line)
        pins[req.name.lower()] = req.specifier
    return pins


def _bounds(spec: SpecifierSet) -> tuple[Version | None, Version | None]:
    """The lowest and highest version a specifier set admits (``==`` and ``~=`` pin both ends)."""
    low = high = None
    for s in spec:
        v = Version(s.version.rstrip(".*"))
        if s.operator in (">=", ">", "==", "~="):
            low = v if low is None else max(low, v)
        if s.operator in ("<", "<=", "==", "~="):
            high = v if high is None else min(high, v)
    return low, high


@pytest.mark.general
@pytest.mark.parametrize("sim", sorted(BACKEND_REQUIREMENTS, key=lambda s: s.value))
def test_the_extra_installs_only_versions_the_policy_supports(sim):
    """`pip install metasim[<backend>]` must not resolve to a version `enforce_backend_versions` then refuses.

    The extra and ``BACKEND_REQUIREMENTS`` are the same promise written twice; this keeps the two in
    step (a new upstream release landed in CI as ``mujoco 3.17.0`` while the policy stopped at 3.14).
    """
    pins = _extra_pins(sim.value)
    if not pins:
        pytest.skip(f"no [{sim.value}] extra")
    for req in BACKEND_REQUIREMENTS[sim]:
        pin = pins.get(req.dist.lower())
        if req.optional or pin is None:
            continue
        policy_low, policy_high = _bounds(SpecifierSet(req.spec))
        pin_low, pin_high = _bounds(pin)
        if policy_high is not None:
            assert pin_high is not None and pin_high <= policy_high, (
                f"[{sim.value}] extra pins {req.dist}{pin or ' (any version)'} but the backend supports {req.spec}"
            )
        if policy_low is not None:
            assert pin_low is not None and pin_low >= policy_low, (
                f"[{sim.value}] extra pins {req.dist}{pin or ' (any version)'} below the supported {req.spec}"
            )
