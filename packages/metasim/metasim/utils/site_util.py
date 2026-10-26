"""World poses of MJCF sites, computed from the state instead of read from a simulator.

A site is a frame fixed to one body of an asset: LIBERO's "contain_region" on a basket, "top_region" on a
cabinet drawer. Its world pose is the pose of that body composed with an offset that never changes, so it can
be computed from :class:`~metasim.types.TensorState` on any backend that reports the body, instead of reading
``site_xpos`` off one backend's physics. The offset comes from the object's MJCF, parsed once.

For a rigid object the site is fixed to the object's root. For an articulation it is fixed to its nearest
named ancestor body, which must appear in ``body_names``.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from metasim.scenario.objects import ArticulationObjCfg
from metasim.utils.math import matrix_from_quat, quat_apply, quat_mul
from metasim.utils.xml_safe import ET

__all__ = ["SitePose", "site_pose"]


@dataclasses.dataclass(frozen=True)
class SitePose:
    """A site's world pose, one row per env."""

    pos: torch.Tensor
    """(N, 3) position."""
    quat: torch.Tensor
    """(N, 4) orientation, wxyz."""
    size: tuple[float, ...] | None = None
    """The site's declared ``size`` (half-extents for a box), scaled per axis like ``pos``; None if unset."""

    @property
    def rot(self) -> torch.Tensor:
        """(N, 3, 3) rotation matrix."""
        return matrix_from_quat(self.quat)


@dataclasses.dataclass(frozen=True)
class _SiteOffset:
    body: str | None  # nearest named ancestor body; None = the asset root
    pos: tuple[float, float, float]  # relative to ``body``
    quat: tuple[float, float, float, float]  # wxyz, relative to ``body``
    root_pos: tuple[float, float, float]  # relative to the asset root
    root_quat: tuple[float, float, float, float]
    size: tuple[float, ...] | None


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _rotate(q, v):
    w, x, y, z = _quat_mul(_quat_mul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def _axis_angle(axis, angle):
    n = math.sqrt(sum(c * c for c in axis))
    s = math.sin(angle / 2) / n
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def _floats(text: str | None, n: int, path: str, tag: str) -> tuple[float, ...] | None:
    if text is None:
        return None
    vals = tuple(float(v) for v in text.split())
    if len(vals) != n:
        raise ValueError(f"{path}: <{tag}> expects {n} numbers, got {text!r}")
    return vals


def _local_frame(el, *, path: str, degrees: bool, eulerseq: str, scale) -> tuple[tuple, tuple]:
    """The frame an MJCF ``<body>`` / ``<site>`` element declares relative to its parent."""
    for unsupported in ("xyaxes", "zaxis", "fromto"):
        if el.get(unsupported) is not None:
            raise ValueError(f"{path}: <{el.tag} {unsupported}=...> is not supported; use pos/quat/euler/axisangle")
    if el.get("class") is not None or el.get("childclass") is not None:
        raise ValueError(f"{path}: <{el.tag} class=...>: default classes are not supported for site offsets")
    pos = _floats(el.get("pos"), 3, path, el.tag) or (0.0, 0.0, 0.0)
    pos = tuple(p * s for p, s in zip(pos, scale, strict=True))
    quat = (1.0, 0.0, 0.0, 0.0)
    if (q := _floats(el.get("quat"), 4, path, el.tag)) is not None:
        n = math.sqrt(sum(c * c for c in q))
        quat = tuple(c / n for c in q)
    elif (aa := _floats(el.get("axisangle"), 4, path, el.tag)) is not None:
        quat = _axis_angle(aa[:3], math.radians(aa[3]) if degrees else aa[3])
    elif (eu := _floats(el.get("euler"), 3, path, el.tag)) is not None:
        for axis_name, angle in zip(eulerseq, eu, strict=True):
            axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis_name.lower()]
            r = _axis_angle(axis, math.radians(angle) if degrees else angle)
            # lower case: axes move with the frame (post-multiply); upper case: fixed axes (pre-multiply)
            quat = _quat_mul(quat, r) if axis_name.islower() else _quat_mul(r, quat)
    return pos, quat


@functools.cache
def _site_offsets(mjcf_path: str, scale: tuple[float, float, float]) -> dict[str, _SiteOffset]:
    root = ET.parse(mjcf_path).getroot()
    for tag in ("include", "frame"):
        if root.find(f".//{tag}") is not None:
            raise ValueError(f"{mjcf_path}: <{tag}> is not supported when computing site offsets")
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") == "degree"
    eulerseq = "xyz" if compiler is None else compiler.get("eulerseq", "xyz")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{mjcf_path}: no <worldbody>")

    offsets: dict[str, _SiteOffset] = {}

    def compose(pos, quat, cpos, cquat):
        return tuple(a + b for a, b in zip(pos, _rotate(quat, cpos), strict=True)), _quat_mul(quat, cquat)

    identity = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def walk(el, anchor: str | None, rel, root):
        for child in el:
            if child.tag not in ("body", "site"):
                continue
            local = _local_frame(child, path=mjcf_path, degrees=degrees, eulerseq=eulerseq, scale=scale)
            child_rel, child_root = compose(*rel, *local), compose(*root, *local)
            if child.tag == "site":
                if (name := child.get("name")) is not None:
                    size = child.get("size")
                    if size is not None:
                        vals = [float(v) for v in size.split()]
                        # a box's half-extents scale per axis; a radius-like size by the largest factor
                        size = (
                            tuple(v * sc for v, sc in zip(vals, scale, strict=True))
                            if len(vals) == 3
                            else tuple(v * max(scale) for v in vals)
                        )
                    offsets[name] = _SiteOffset(anchor, *child_rel, *child_root, size)
            elif child.get("name"):
                walk(child, child.get("name"), identity, child_root)
            else:
                walk(child, anchor, child_rel, child_root)

    walk(worldbody, None, identity, identity)
    return offsets


def site_pose(states, obj, site: str) -> SitePose:
    """World pose of ``site`` on object ``obj`` (its config), from ``states``.

    Args:
        states: the :class:`~metasim.types.TensorState` the pose is read from.
        obj: the object's config; its ``mjcf_path`` declares the site.
        site: the site's name in that MJCF, without the object prefix.

    Raises:
        KeyError: the MJCF declares no such site, or an articulation's state does not report the body
            the site is fixed to.
    """
    if obj.mjcf_path is None:
        raise ValueError(f"site_pose: object {obj.name!r} has no mjcf_path to read site {site!r} from")
    scale = tuple(getattr(obj, "scale", (1.0, 1.0, 1.0)))
    offsets = _site_offsets(obj.mjcf_path, scale)
    if site not in offsets:
        raise KeyError(f"site_pose: {obj.mjcf_path} declares no site {site!r}; it has {sorted(offsets)}")
    off = offsets[site]
    state = states.objects[obj.name]
    frame, pos, quat = state.root_state[:, :7], off.root_pos, off.root_quat
    if off.body is not None and isinstance(obj, ArticulationObjCfg):
        names = list(state.body_names or [])
        if off.body not in names:
            raise KeyError(
                f"site_pose: site {site!r} is fixed to body {off.body!r} of {obj.name!r}, which its state does "
                f"not report (body_names={names})"
            )
        frame, pos, quat = state.body_state[:, names.index(off.body), :7], off.pos, off.quat
    n = frame.shape[0]
    local_pos = torch.tensor(pos, dtype=frame.dtype, device=frame.device).expand(n, 3)
    local_quat = torch.tensor(quat, dtype=frame.dtype, device=frame.device).expand(n, 4)
    return SitePose(
        pos=frame[:, :3] + quat_apply(frame[:, 3:7], local_pos),
        quat=quat_mul(frame[:, 3:7], local_quat),
        size=off.size,
    )
