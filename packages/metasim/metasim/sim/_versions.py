"""Backend version policy: what each simulator backend requires, what it was last verified with.

Simulator packages move fast and break APIs between minor releases (newton 1.2 → 1.5 → 1.6 each
renamed core fields; MuJoCo changed its arena defaults; SuperDex is pre-1.0 in spirit). Instead of
discovering that at the first ``AttributeError`` deep inside a handler, every backend declares:

* ``spec`` — the version range the handler's code paths are written for. Outside it, creating the
  handler raises :class:`BackendVersionError` (``METASIM_SKIP_VERSION_CHECK=1`` overrides).
* ``tested`` — the exact version the backend's test suite last passed with. A different version
  inside ``spec`` only warns once, so the log says "untested" before anything else goes wrong.

``python -m metasim doctor`` prints the table for every backend; the weekly ``backend-compat``
workflow installs the newest release of each CPU-installable backend and runs its suite, so
``tested`` is bumped deliberately, not by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from loguru import logger as log
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from metasim.constants import SimType


class BackendVersionError(RuntimeError):
    """An installed simulator package is outside the range the backend supports."""


@dataclass(frozen=True)
class Requirement:
    """One distribution a backend depends on."""

    dist: str
    spec: str
    tested: str = ""
    optional: bool = False
    """Optional distributions (viewers, renderers) are reported but never block handler creation."""


@dataclass
class RequirementStatus:
    """The installed state of one :class:`Requirement`."""

    requirement: Requirement
    installed: str | None
    in_spec: bool | None  # None when not installed
    is_tested: bool | None

    @property
    def label(self) -> str:
        if self.installed is None:
            return "missing" if not self.requirement.optional else "absent (optional)"
        if not self.in_spec:
            return "UNSUPPORTED"
        return "ok" if self.is_tested or not self.requirement.tested else "untested"


@dataclass
class BackendVersionReport:
    """Every requirement of one backend, with a verdict."""

    sim: SimType
    statuses: list[RequirementStatus] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        required = [s for s in self.statuses if not s.requirement.optional]
        return bool(required) and all(s.installed is not None for s in required)

    @property
    def unsupported(self) -> list[RequirementStatus]:
        return [
            s for s in self.statuses if s.installed is not None and s.in_spec is False and not s.requirement.optional
        ]

    @property
    def untested(self) -> list[RequirementStatus]:
        return [
            s
            for s in self.statuses
            if s.installed is not None and s.in_spec and s.requirement.tested and not s.is_tested
        ]


# Ranges are the versions the handler code paths target (see the compat modules under
# metasim/sim/<backend>/); ``tested`` is the last version the backend suite passed with. Keep both
# honest: widen ``spec`` only with a passing run, bump ``tested`` from the backend-compat workflow.
BACKEND_REQUIREMENTS: dict[SimType, tuple[Requirement, ...]] = {
    SimType.MUJOCO: (
        Requirement("mujoco", ">=3.2,<3.14", tested="3.12.0"),
        Requirement("dm-control", ">=1.0.20,<2", tested="1.0.45"),
    ),
    SimType.MJX: (
        Requirement("mujoco", ">=3.2,<3.14", tested="3.12.0"),
        Requirement("mujoco-mjx", ">=3.2.7,<3.14"),
        Requirement("jax", ">=0.4.30,<0.8"),
    ),
    SimType.NEWTON: (
        Requirement("newton", ">=1.5,<2", tested="1.6.0.dev0"),
        Requirement("warp-lang", ">=1.11,<2", tested="1.17.0"),
        Requirement("mujoco-warp", ">=0.0.1", tested="3.12.0"),
        Requirement("mujoco", ">=3.3.7,<3.14", tested="3.12.0"),
    ),
    SimType.SUPERDEX: (
        Requirement("superdex-physics", ">=1.0,<2", tested="1.0.0"),
        Requirement("superdex-robotics", ">=1.0,<2", tested="1.0.0"),
        Requirement("pyrender", ">=0.1.45", optional=True),
    ),
    SimType.SAPIEN3: (Requirement("sapien", ">=3.0.0b1,<4"),),
    SimType.SAPIEN2: (Requirement("sapien", ">=2.2,<3"),),
    SimType.PYBULLET: (Requirement("pybullet", ">=3.2,<4"),),
    SimType.GENESIS: (Requirement("genesis-world", ">=0.2,<1"),),
    SimType.ISAACSIM: (
        Requirement("isaacsim", ">=4.5,<5.1", tested="5.0.0.0"),
        # The handler is built on Isaac Lab (AppLauncher, sim utils); it is not on PyPI, so the
        # ``isaacsim`` extra cannot pull it — see requirements/isaacsim5.txt.
        # Isaac Lab tags vs the `isaaclab` distribution: v2.0.2 -> 0.34.9, v2.1.0 -> 0.36.21,
        # v2.1.1 -> 0.41.3, v2.2.1 -> 0.45.9. v2.0+ pairs with Isaac Sim 4.5+, hence the floor.
        Requirement("isaaclab", ">=0.34,<1", tested="0.45.9"),
    ),
    SimType.ISAACGYM: (Requirement("isaacgym", ">=1.0rc4"),),
    SimType.BLENDER: (Requirement("bpy", ">=4.0,<5"),),
}


def _installed(dist: str) -> str | None:
    try:
        return dist_version(dist)
    except PackageNotFoundError:
        return None


def _status(req: Requirement) -> RequirementStatus:
    installed = _installed(req.dist)
    if installed is None:
        return RequirementStatus(req, None, None, None)
    try:
        in_spec = Version(installed) in SpecifierSet(req.spec, prereleases=True)
    except InvalidVersion:
        in_spec = False
    is_tested = (installed == req.tested) if req.tested else None
    return RequirementStatus(req, installed, in_spec, is_tested)


def check_backend(sim: SimType) -> BackendVersionReport:
    """Inspect the packages backing ``sim`` without importing them."""
    return BackendVersionReport(sim, [_status(r) for r in BACKEND_REQUIREMENTS.get(sim, ())])


_WARNED: set[SimType] = set()


def enforce_backend_versions(sim: SimType) -> BackendVersionReport:
    """Called before a handler class is imported: raise on an unsupported version, warn once on an
    untested one. ``METASIM_SKIP_VERSION_CHECK=1`` turns the error into a warning.
    """
    report = check_backend(sim)
    if report.unsupported:
        lines = ", ".join(
            f"{s.requirement.dist} {s.installed} (supported: {s.requirement.spec})" for s in report.unsupported
        )
        message = (
            f"{sim.value}: installed simulator packages are outside the range this backend supports: {lines}. "
            f"Install a supported version, or set METASIM_SKIP_VERSION_CHECK=1 to proceed at your own risk."
        )
        if os.environ.get("METASIM_SKIP_VERSION_CHECK") == "1":
            log.warning(message)
        else:
            raise BackendVersionError(message)
    if report.untested and sim not in _WARNED:
        _WARNED.add(sim)
        lines = ", ".join(
            f"{s.requirement.dist} {s.installed} (last verified: {s.requirement.tested})" for s in report.untested
        )
        log.warning(
            f"{sim.value}: running on a simulator version the test suite has not been run against: {lines}. "
            f"`python -m metasim doctor` shows the full table; report breakage with these versions."
        )
    return report


def doctor(sims: list[SimType] | None = None) -> list[BackendVersionReport]:
    """Reports for every backend (or ``sims``)."""
    return [check_backend(s) for s in (sims or list(BACKEND_REQUIREMENTS))]


def format_reports(reports: list[BackendVersionReport]) -> str:
    """A fixed-width table for terminals."""
    rows = [("backend", "package", "installed", "supported", "last verified", "status")]
    for rep in reports:
        for st in rep.statuses:
            rows.append((
                rep.sim.value,
                st.requirement.dist,
                st.installed or "-",
                st.requirement.spec,
                st.requirement.tested or "-",
                st.label,
            ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)
