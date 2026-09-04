"""A self-describing, replayable episode record.

An episode recorded on one machine must be replayable on another. That needs three things the
legacy demo formats do not have: the **full physical state** at every step (positions *and*
velocities, joint targets, body poses) in a lossless numeric form, the **names** that give those
numbers meaning (joint and body order, quaternion convention, root-state layout), and the
**provenance** that produced them (simulator backend and its installed versions, MetaSim version
and git commit, physics step and decimation, seed, the assets by path and content hash, the
platform). :class:`EpisodeRecord` carries all three; :func:`save_episode` / :func:`load_episode`
write and read it as one ``.npz`` file (float64 arrays plus a JSON header, no pickle), and the
loader validates the header and the array shapes before handing anything back.

The record is deliberately independent of any task: it is what a handler saw. Task-level fields
(success, language instruction) go in :attr:`EpisodeRecord.info`.

Conventions, stated in every file so a reader never has to guess:

- ``root_state`` rows are ``[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]`` in the world frame
  (quaternion **wxyz**, velocities of the root link).
- joint arrays follow ``joint_names[entity]`` (the handler's sorted-name order, which is what
  ``get_states`` / ``set_states`` use); body arrays follow ``body_names[entity]``.
- ``states[t]`` is the state *before* ``actions[t]``; ``states[-1]`` is the final state, so a record
  of ``T`` actions holds ``T + 1`` states.
- ``actions`` are the tensors handed to ``set_dof_targets`` (shape ``(T, num_envs, dof)``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger as log

from metasim.types import ObjectState, RobotState, TensorState


class EpisodeFileError(RuntimeError):
    """A roboverse.episode file was handed to a loader that cannot use it.

    Deliberately not a ``ValueError``: the task bases swallow ``ValueError`` from ``get_traj`` as
    "no demo, use the defaults", which would hide the actionable message.
    """


FORMAT = "roboverse.episode"
FORMAT_VERSION = 1
QUATERNION_CONVENTION = "wxyz"
ROOT_STATE_LAYOUT = ("pos_xyz", "quat_wxyz", "lin_vel_world", "ang_vel_world")

_ENTITY_FIELDS = (
    "root_state",
    "body_state",
    "joint_pos",
    "joint_vel",
    "joint_pos_target",
    "joint_vel_target",
    "joint_effort_target",
)
_ASSET_PATH_SUFFIXES = ("_path", "_file")


@dataclass
class Provenance:
    """Everything a replayer needs to know about where an episode came from."""

    simulator: str
    num_envs: int
    dt: float | None
    """``ScenarioCfg.sim_params.dt`` as configured (None = the backend's default)."""
    physics_dt: float | None
    """The step the backend actually used, when it exposes it; None when unknown."""
    decimation: int
    env_step_s: float | None
    """Simulated seconds per ``simulate()`` call (``physics_dt * decimation``) when known."""
    seed: int | None = None
    control_mode: str | None = None
    backend_versions: dict[str, str] = field(default_factory=dict)
    metasim_version: str = ""
    git_commit: str | None = None
    git_dirty: bool | None = None
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    """``entity -> {field: {"path", "sha256", "bytes"}}`` for every asset file the config points at."""
    python: str = ""
    torch: str = ""
    numpy: str = ""
    platform: str = ""
    device: str = ""
    created_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """A recorded episode: names, provenance, ``T + 1`` states and ``T`` actions."""

    provenance: Provenance
    joint_names: dict[str, list[str]]
    body_names: dict[str, list[str]]
    entities: dict[str, list[str]]
    """``{"robots": [...], "objects": [...]}`` in the order the scenario declared them."""
    states: list[TensorState]
    actions: list[torch.Tensor]
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Camera configs (name -> jsonable dict), so intrinsics and poses travel with the data."""
    scenario: dict[str, Any] = field(default_factory=dict)
    """The scenario config as a jsonable dict (best effort; asset paths are also hashed in provenance)."""
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    def validate(self) -> None:
        """Raise ``ValueError`` on any internal inconsistency (called by the loader)."""
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(f"episode has {len(self.states)} states for {len(self.actions)} actions; expected T + 1")
        n = self.provenance.num_envs
        for kind in ("robots", "objects"):
            for name in self.entities.get(kind, []):
                for t, state in enumerate(self.states):
                    entity = getattr(state, kind).get(name)
                    if entity is None:
                        raise ValueError(f"state {t} lacks {kind[:-1]} {name!r}")
                    if tuple(entity.root_state.shape) != (n, 13):
                        raise ValueError(
                            f"{name}.root_state at step {t} has shape {tuple(entity.root_state.shape)}, expected ({n}, 13)"
                        )
                    q = getattr(entity, "joint_pos", None)
                    names = self.joint_names.get(name)
                    if q is not None and names is not None and q.shape[-1] != len(names):
                        raise ValueError(f"{name}.joint_pos has {q.shape[-1]} joints, joint_names lists {len(names)}")
                    quat_norm = torch.linalg.norm(entity.root_state[:, 3:7], dim=-1)
                    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-3):
                        raise ValueError(
                            f"{name}.root_state quaternion at step {t} is not unit length: {quat_norm.tolist()}"
                        )
        for t, action in enumerate(self.actions):
            if action.shape[0] != n:
                raise ValueError(f"action {t} has {action.shape[0]} rows, expected num_envs={n}")


# ----------------------------------------------------------------------------- provenance


def _git_state(start: Path) -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` of the repository containing ``start``; ``(None, None)`` outside a checkout."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(start), "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
        return commit, bool(status.strip())
    except Exception:
        return None, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(value: str) -> Path | None:
    """The asset file a config path names on this machine, by ``hf_util``'s rule.

    Relative to the CWD first; else under ``ROBOVERSE_DATA_DIR``, which *replaces* the
    ``roboverse_data/`` root (``roboverse_data/robots/x`` lives at ``$ROBOVERSE_DATA_DIR/robots/x``).
    """
    path = Path(value).expanduser()
    if path.is_file():
        return path
    data_dir = os.environ.get("ROBOVERSE_DATA_DIR")
    if data_dir and not path.is_absolute():
        parts = path.parts
        rel = Path(*parts[1:]) if parts and parts[0] == "roboverse_data" else path
        candidate = Path(data_dir).expanduser() / rel
        if candidate.is_file():
            return candidate
    return None


def _asset_files(cfg, simulator: str) -> tuple[dict[str, Path], list[str]]:
    """``(label -> file, unresolved)`` for the assets the backend loads for ``cfg``.

    The primary asset is what ``cfg.file_name(simulator)`` names (the per-backend MJCF / URDF / USD
    choice every handler makes); ``extra_resources`` entries are included too. Other ``*_path``
    fields are files this backend never opens, so they are not hashed: a recording must not blame an
    asset the replay never read. Configs without ``file_name`` fall back to their path fields.
    ``unresolved`` lists the configured paths that name no file on this machine; a primitive shape
    configures none and is not unresolved.
    """
    found: dict[str, Path] = {}
    unresolved: list[str] = []

    def take(label: str, value) -> None:
        if not isinstance(value, str) or not value:
            return
        resolved = _resolve_asset(value)
        if resolved is None:
            unresolved.append(value)
        else:
            found[label] = resolved

    primary = None
    file_name = getattr(cfg, "file_name", None)
    if callable(file_name):
        try:
            primary = file_name(simulator)
        except Exception:
            primary = None
    if not isinstance(primary, str) or not primary:
        for name in dir(cfg):
            if not name.startswith("_") and name.endswith(_ASSET_PATH_SUFFIXES):
                take(name, getattr(cfg, name, None))
        return found, unresolved
    take("asset", primary)
    for extra in getattr(cfg, "extra_resources", None) or []:
        take(f"extra:{Path(extra).name}" if isinstance(extra, str) else "", extra)
    return found, unresolved


def _jsonable(value: Any) -> Any:
    """A JSON-serialisable view of a config / value (tensors and arrays become lists, objects dicts)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and hasattr(value, "value"):  # enums
        return value.value if isinstance(value.value, (str, int, float)) else str(value)
    if hasattr(value, "__dict__"):  # plain objects / namespaces: their public attributes
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)


def _backend_versions(simulator: str) -> dict[str, str]:
    try:
        from metasim.constants import SimType
        from metasim.sim._versions import check_backend

        report = check_backend(SimType(simulator))
        return {s.requirement.dist: s.installed for s in report.statuses if s.installed is not None}
    except Exception:
        return {}


def _resolved_physics_dt(handler) -> float | None:
    """``handler.physics_dt`` (the backend contract); None when the backend has not resolved it."""
    try:
        value = handler.physics_dt
    except Exception:
        return None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def env_step_seconds(handler, physics_dt: float | None) -> float | None:
    """``handler.env_step_s`` (the backend contract, forwarded by the wrappers); None when unknown."""
    try:
        value = handler.env_step_s
    except Exception:
        return None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def provenance_from_handler(
    handler, *, seed: int | None = None, extras: dict[str, Any] | None = None, num_envs: int | None = None
) -> Provenance:
    """Collect the provenance of a launched handler: backend, versions, time base, assets, platform."""
    import metasim

    scenario = handler.scenario
    sim_params = getattr(scenario, "sim_params", None)
    dt = getattr(sim_params, "dt", None)
    physics_dt = _resolved_physics_dt(handler)
    decimation = int(getattr(scenario, "decimation", 1))
    env_step = env_step_seconds(handler, physics_dt)
    assets: dict[str, dict[str, Any]] = {}
    entities = list(getattr(scenario, "robots", []) or []) + list(getattr(scenario, "objects", []) or [])
    scene = getattr(scenario, "scene", None)
    if scene is not None and not isinstance(scene, str):
        entities.append(scene)  # loaded like any other asset (MuJoCo builds the root model from it)
    for cfg in entities:
        files, unresolved = _asset_files(cfg, str(getattr(scenario, "simulator", "")))
        label = getattr(cfg, "name", type(cfg).__name__)
        if unresolved:
            log.warning(f"provenance: asset(s) of {label!r} not found on this machine, not hashed: {unresolved}")
        if files:
            assets[label] = {
                f: {"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size} for f, p in files.items()
            }
    commit, dirty = _git_state(Path(metasim.__file__).resolve().parent)
    device = getattr(handler, "device", None)
    return Provenance(
        simulator=str(getattr(scenario, "simulator", "")),
        num_envs=int(
            num_envs if num_envs is not None else getattr(handler, "num_envs", getattr(scenario, "num_envs", 1))
        ),
        dt=float(dt) if dt is not None else None,
        physics_dt=physics_dt,
        decimation=decimation,
        env_step_s=env_step,
        seed=seed,
        control_mode=getattr(sim_params, "superdex_control_mode", None)
        if str(scenario.simulator) == "superdex"
        else None,
        backend_versions=_backend_versions(str(scenario.simulator)),
        metasim_version=str(getattr(metasim, "__version__", "")),
        git_commit=commit,
        git_dirty=dirty,
        assets=assets,
        python=sys.version.split()[0],
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        device=str(device) if device is not None else "",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extras=dict(extras or {}),
    )


# ----------------------------------------------------------------------------- record


def record_episode(
    handler,
    initial_state: TensorState,
    actions: list[torch.Tensor],
    *,
    seed: int | None = None,
    info: dict[str, Any] | None = None,
) -> EpisodeRecord:
    """Write ``initial_state``, apply ``actions`` one per ``simulate()``, and return the full record."""
    from metasim.utils.replay import record

    if seed is not None:
        handler.set_seed(seed)
    traj = record(handler, initial_state, list(actions))
    return episode_from_states(handler, traj.states, traj.actions, seed=seed, info=info)


def episode_from_states(
    handler,
    states: list[TensorState],
    actions: list[torch.Tensor],
    *,
    seed: int | None = None,
    info: dict[str, Any] | None = None,
    num_envs: int | None = None,
    provenance: Provenance | None = None,
) -> EpisodeRecord:
    """Build a record from states a caller already captured (``states[t]`` precedes ``actions[t]``).

    ``num_envs`` overrides the handler's when the states are a per-env slice (one episode of a batch).
    ``provenance`` lets a collector compute the run-invariant, asset-hashing provenance once and reuse
    it per episode; ``seed`` and ``num_envs`` are stamped onto a copy.
    """
    scenario = handler.scenario
    robots = [r.name for r in scenario.robots]
    objects = [o.name for o in scenario.objects]
    joint_names = {name: list(handler.get_joint_names(name, sort=True)) for name in robots + objects}
    joint_names = {k: v for k, v in joint_names.items() if v}
    body_names: dict[str, list[str]] = {}
    for name in robots + objects:
        try:
            names = list(handler.get_body_names(name, sort=True))
        except Exception:
            names = []
        if names:
            body_names[name] = names
    cameras = {cam.name: _jsonable(cam) for cam in getattr(scenario, "cameras", []) or []}
    if provenance is None:
        provenance = provenance_from_handler(handler, seed=seed, num_envs=num_envs)
    else:
        provenance = dataclasses.replace(
            provenance,
            seed=seed if seed is not None else provenance.seed,
            num_envs=int(num_envs) if num_envs is not None else provenance.num_envs,
        )
    return EpisodeRecord(
        provenance=provenance,
        joint_names=joint_names,
        body_names=body_names,
        entities={"robots": robots, "objects": objects},
        states=[_detach(s) for s in states],
        actions=[a.detach().cpu().double() for a in actions],
        cameras=cameras,
        scenario=_jsonable(scenario),
        info=dict(info or {}),
    )


def _detach(state: TensorState) -> TensorState:
    """Physical fields of ``state`` as float64 CPU tensors (camera images are not part of a record)."""

    def entity(st):
        kwargs = {}
        for f in dataclasses.fields(st):
            value = getattr(st, f.name)
            kwargs[f.name] = value.detach().cpu().double() if isinstance(value, torch.Tensor) else value
        return type(st)(**kwargs)

    return TensorState(
        objects={k: entity(v) for k, v in state.objects.items()},
        robots={k: entity(v) for k, v in state.robots.items()},
        cameras={},
        extras={k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in (state.extras or {}).items()},
    )


# ----------------------------------------------------------------------------- save / load


def save_episode(record: EpisodeRecord, path: str | Path) -> Path:
    """Write ``record`` to ``path`` (``.npz``: float64 arrays + a JSON header; no pickle)."""
    record.validate()
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz") if path.suffix else path.with_suffix(".npz")
    arrays: dict[str, np.ndarray] = {}
    for kind in ("robots", "objects"):
        for name in record.entities[kind]:
            for f in _ENTITY_FIELDS:
                stack = [getattr(getattr(s, kind)[name], f, None) for s in record.states]
                if any(v is None for v in stack):
                    continue
                arrays[f"{kind}/{name}/{f}"] = np.stack([v.numpy() for v in stack]).astype(np.float64)
    arrays["actions"] = (
        np.stack([a.numpy() for a in record.actions]).astype(np.float64) if record.actions else np.zeros((0,))
    )
    header = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "quaternion": QUATERNION_CONVENTION,
        "root_state_layout": list(ROOT_STATE_LAYOUT),
        "num_steps": len(record.actions),
        "entities": record.entities,
        "joint_names": record.joint_names,
        "body_names": record.body_names,
        "provenance": dataclasses.asdict(record.provenance),
        "cameras": record.cameras,
        "scenario": record.scenario,
        "info": record.info,
        "arrays": sorted(arrays),
    }
    arrays["header"] = np.array(json.dumps(header, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def read_header(path: str | Path) -> dict[str, Any]:
    """The JSON header of an episode file (format, provenance, names) without loading the arrays."""
    with np.load(Path(path), allow_pickle=False) as data:
        if "header" not in data:
            raise ValueError(f"{path} is not a {FORMAT} file (no header)")
        header = json.loads(str(data["header"]))
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: format {header.get('format')!r} is not {FORMAT!r}")
    if int(header.get("format_version", -1)) > FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version {header['format_version']} is newer than this reader ({FORMAT_VERSION}); upgrade metasim"
        )
    return header


def is_episode_file(path: str | Path) -> bool:
    """True when ``path`` is an episode file of this format (by content, not by name)."""
    try:
        read_header(path)
        return True
    except Exception:
        return False


def load_episode(path: str | Path) -> EpisodeRecord:
    """Read and validate an episode file."""
    header = read_header(path)
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {k: torch.from_numpy(np.asarray(data[k])) for k in data.files if k != "header"}
    num_steps = int(header["num_steps"])
    entities = header["entities"]
    states = []
    for t in range(num_steps + 1):
        robots, objects = {}, {}
        for kind, target, cls in (("robots", robots, RobotState), ("objects", objects, ObjectState)):
            for name in entities[kind]:
                kwargs: dict[str, Any] = {}
                for f in _ENTITY_FIELDS:
                    key = f"{kind}/{name}/{f}"
                    if key in arrays and any(fl.name == f for fl in dataclasses.fields(cls)):
                        kwargs[f] = arrays[key][t]
                if "root_state" not in kwargs:
                    raise ValueError(f"{path}: {kind[:-1]} {name!r} has no root_state array")
                names = header["body_names"].get(name)
                if names and any(fl.name == "body_names" for fl in dataclasses.fields(cls)):
                    kwargs["body_names"] = list(names)
                target[name] = cls(**kwargs)
        states.append(TensorState(objects=objects, robots=robots, cameras={}, extras={}))
    actions = [arrays["actions"][t] for t in range(num_steps)] if num_steps else []
    provenance = Provenance(**header["provenance"])
    record = EpisodeRecord(
        provenance=provenance,
        joint_names={k: list(v) for k, v in header["joint_names"].items()},
        body_names={k: list(v) for k, v in header["body_names"].items()},
        entities={k: list(v) for k, v in entities.items()},
        states=states,
        actions=actions,
        cameras=header.get("cameras", {}),
        scenario=header.get("scenario", {}),
        info=header.get("info", {}),
    )
    record.validate()
    return record


def check_assets(record: EpisodeRecord) -> dict[str, str]:
    """Compare the recorded asset hashes with the files on this machine: ``field -> "ok" | "missing" | "changed"``."""
    result: dict[str, str] = {}
    for entity, files in record.provenance.assets.items():
        for f, meta in files.items():
            found = _resolve_asset(meta["path"])
            if found is None:
                result[f"{entity}.{f}"] = "missing"
            else:
                result[f"{entity}.{f}"] = "ok" if _sha256(found) == meta["sha256"] else "changed"
    return result
