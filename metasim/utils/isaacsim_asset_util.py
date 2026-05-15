"""Helpers for resolving IsaacSim-loadable asset paths."""

from __future__ import annotations

import os
import tempfile
from metasim.utils.xml_safe import ET  # defused parser for untrusted MJCF/URDF
from pathlib import Path

from loguru import logger as log


def _patch_urdf_mesh_paths_to_absolute(urdf_path: Path) -> str:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if os.path.isabs(filename) and os.path.exists(filename):
            continue

        candidates: list[Path] = []
        if filename.startswith("package://"):
            rel_path = filename.split("package://", 1)[1]
            candidates.append(urdf_path.parent / rel_path)
        candidates.extend([
            urdf_path.parent / filename,
            urdf_path.parent / "meshes" / filename,
            urdf_path.parent / "visual" / filename,
            urdf_path.parent / "collision" / filename,
            urdf_path.parent.parent / "meshes" / filename,
        ])

        resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            log.warning(f"Could not resolve URDF mesh path for IsaacSim conversion: {filename}")
            continue
        mesh.set("filename", str(resolved))
        changed = True

    if not changed:
        return str(urdf_path)

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf")
    tree.write(handle.name)
    return handle.name


def convert_urdf_to_usd_cached(urdf_path: str) -> str:
    """Convert a URDF asset to USD once and reuse the generated file.

    The generated USD is placed next to the URDF, matching the existing
    SceneRandomizer convention.
    """
    urdf_path_obj = Path(urdf_path).expanduser().resolve()
    if not urdf_path_obj.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path_obj}")

    usd_output = urdf_path_obj.with_suffix(".usd")
    if usd_output.exists():
        return str(usd_output)

    log.info(f"Converting URDF to USD: {urdf_path_obj}")
    patched_urdf_path = _patch_urdf_mesh_paths_to_absolute(urdf_path_obj)

    try:
        try:
            from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
        except ImportError:
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except Exception as exc:  # pragma: no cover - depends on IsaacLab runtime
        raise RuntimeError("IsaacSim URDF fallback requires IsaacLab UrdfConverter support") from exc

    patched_path_obj = Path(patched_urdf_path)
    try:
        cfg = UrdfConverterCfg(
            asset_path=str(patched_path_obj),
            usd_dir=os.path.abspath(str(usd_output.parent)),
            usd_file_name=usd_output.name,
            fix_base=False,
            force_usd_conversion=True,
            make_instanceable=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        )
        converter = UrdfConverter(cfg)
        generated_path = Path(converter.usd_path).expanduser().resolve()
    finally:
        if patched_path_obj != urdf_path_obj and patched_path_obj.exists():
            patched_path_obj.unlink(missing_ok=True)

    if not generated_path.exists():
        raise RuntimeError(f"URDF conversion did not produce USD output: {generated_path}")
    return str(generated_path)


def convert_mjcf_to_usd_cached(mjcf_path: str) -> str:
    """Convert an MJCF asset to USD once and reuse the generated file.

    Mirrors ``convert_urdf_to_usd_cached`` so cfgs that ship only an MJCF
    (no URDF, no pre-built USD) still resolve under IsaacSim.
    """
    mjcf_path_obj = Path(mjcf_path).expanduser().resolve()
    if not mjcf_path_obj.is_file():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path_obj}")

    usd_output = mjcf_path_obj.with_suffix(".usd")
    if usd_output.exists():
        return str(usd_output)

    log.info(f"Converting MJCF to USD: {mjcf_path_obj}")

    try:
        try:
            from omni.isaac.lab.sim.converters import MjcfConverter, MjcfConverterCfg
        except ImportError:
            from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    except Exception as exc:  # pragma: no cover - depends on IsaacLab runtime
        raise RuntimeError("IsaacSim MJCF fallback requires IsaacLab MjcfConverter support") from exc

    cfg = MjcfConverterCfg(
        asset_path=str(mjcf_path_obj),
        usd_dir=os.path.abspath(str(usd_output.parent)),
        usd_file_name=usd_output.name,
        fix_base=False,
        force_usd_conversion=True,
        make_instanceable=True,
    )
    converter = MjcfConverter(cfg)
    generated_path = Path(converter.usd_path).expanduser().resolve()

    if not generated_path.exists():
        raise RuntimeError(f"MJCF conversion did not produce USD output: {generated_path}")
    return str(generated_path)


def _existing_path(maybe_path) -> Path | None:
    if not maybe_path:
        return None
    p = Path(os.path.expanduser(str(maybe_path))).resolve()
    return p if p.is_file() else None


def resolve_isaacsim_file_path(cfg) -> str:
    """Resolve the concrete file path IsaacSim should load for a config object.

    Looks for an existing USD first; if the configured ``usd_path`` is
    missing on disk, falls through to URDF and then MJCF, converting
    each to USD on demand. ``cfg`` is treated as a duck type — anything
    with ``usd_path`` / ``urdf_path`` / ``mjcf_path`` attributes works.
    """
    existing_usd = _existing_path(getattr(cfg, "usd_path", None))
    if existing_usd is not None:
        return str(existing_usd)

    urdf = _existing_path(getattr(cfg, "urdf_path", None))
    if urdf is not None:
        return convert_urdf_to_usd_cached(str(urdf))

    mjcf = _existing_path(getattr(cfg, "mjcf_path", None))
    if mjcf is not None:
        return convert_mjcf_to_usd_cached(str(mjcf))

    name = getattr(cfg, "name", "<unnamed>")
    raise ValueError(
        f"{type(cfg).__name__} '{name}' requires an existing usd_path, "
        "urdf_path, or mjcf_path for isaacsim"
    )
