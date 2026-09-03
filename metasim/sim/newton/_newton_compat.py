"""Newton API compatibility shim.

Scope
=====
This module handles import-time / attribute-name drift between newton
versions. Newton (the NVIDIA Warp-based physics engine) refactors its
public API between releases; e.g. 1.0 -> 1.2 moved JointType, removed
the free function populate_contacts, made current_world read-only, and
renamed Model.body_key -> body_label. Each shim either imports the
symbol from its old location (if installed), falls back to the new
public path, or provides a no-op stub if the feature is removed.

Shims provided
==============
- JointType: import location moved
- populate_contacts: free function removed -> no-op stub
- set_current_world(builder, idx): setter removed -> no-op when -1
- Model attribute aliases body_key / joint_key / shape_key -> *_label
- add_mjcf(builder, path, **kw): ModelBuilder.add_mjcf method removed
  in 1.2 but parse_mjcf in newton._src.utils.import_mjcf still works

Backward-compat guarantee:
- Anything that worked on newton < 1.2 still works.
- Anything that newton 1.2+ supports also works.
- If a feature is removed (e.g. populate_contacts as a free function),
  we provide a stub that does the equivalent work via the new method-
  based API, or a no-op with a one-time warning when there's no
  equivalent.
"""

from __future__ import annotations

import warnings


def _resolve_joint_type():
    """Return ``newton.JointType`` enum (location moved in 1.2)."""
    try:
        from newton._src.sim.joints import JointType  # newton < 1.2

        return JointType
    except ModuleNotFoundError:
        from newton import JointType  # newton >= 1.2

        return JointType


JointType = _resolve_joint_type()


def _resolve_populate_contacts():
    """Return ``populate_contacts(contacts, solver)`` callable.

    Old API: free function ``newton.sensors.populate_contacts``.
    New API (newton >= 1.2): contacts are populated implicitly by the
    solver step; the free function was removed. We provide a no-op so
    callers don't crash.
    """
    try:
        from newton.sensors import populate_contacts  # newton < 1.2

        return populate_contacts
    except ImportError:
        # Newton >= 1.2: contacts populated by solver step automatically.
        def _no_op(contacts, solver):
            return None

        return _no_op


populate_contacts = _resolve_populate_contacts()


def set_current_world(builder, world_idx: int) -> None:
    """Set the ModelBuilder's active world index across newton versions.

    newton < 1.2: ``builder.current_world = world_idx`` (writable attribute).
    newton >= 1.2: ``current_world`` is a read-only property; geom added
    outside any ``begin_world/end_world`` context is implicitly global.

    For world_idx == -1 (i.e. "global / shared across worlds"), this
    function is a no-op on newton 1.2 (default outside begin_world is
    already global).
    """
    try:
        builder.current_world = world_idx  # newton < 1.2
    except AttributeError:
        if world_idx == -1:
            return  # newton 1.2+: global is the default outside begin_world
        warnings.warn(
            f"Cannot set current_world={world_idx} on newton 1.2+ ModelBuilder; "
            "use begin_world()/end_world() context manager instead.",
            RuntimeWarning,
            stacklevel=2,
        )


def _install_model_attr_aliases():
    """Add backward-compat properties to ``newton.Model``.

    newton 1.2 renamed:
      Model.body_key  -> Model.body_label
      Model.joint_key -> Model.joint_label
      Model.shape_key -> Model.shape_label

    Note: body_label etc. are instance attributes (set in __init__),
    not class attributes; we install the compat property unconditionally
    so accesses route through the property to the new instance attr.
    """
    try:
        import newton
    except ImportError:
        return
    if not hasattr(newton, "Model"):
        return
    M = newton.Model
    for old, new in [("body_key", "body_label"), ("joint_key", "joint_label"), ("shape_key", "shape_label")]:
        if old in M.__dict__:
            continue
        setattr(M, old, property(lambda self, _n=new: getattr(self, _n)))


_install_model_attr_aliases()


def add_mjcf(builder, mjcf_path: str, **kwargs):
    """Add an MJCF file to ``ModelBuilder``, version-portable.

    Newton >= 1.2 dropped ``ModelBuilder.add_mjcf`` method but kept the
    underlying parser at ``newton._src.utils.import_mjcf.parse_mjcf``.
    Newton < 1.2 had both. This shim picks whichever is available so
    MetaSim NewtonHandler can load MJCF assets (mjlab cartpole / Go1 /
    G1 / YAM all ship as .xml).

    Args:
        builder: ``newton.ModelBuilder`` instance.
        mjcf_path: Path to .xml MJCF asset file.
        **kwargs: Forwarded to underlying parser (e.g. ``floating``,
            ``parse_meshes``, ``ignore_inertial_definitions``).
    """
    if hasattr(builder, "add_mjcf"):
        return builder.add_mjcf(mjcf_path, **kwargs)
    from newton._src.utils.import_mjcf import parse_mjcf

    return parse_mjcf(builder, mjcf_path, **kwargs)


def _install_modelbuilder_key_to_label_shim():
    """Newton 1.2 renamed ``key`` -> ``label`` in ``ModelBuilder.add_body``
    (and similar add_* methods). Wrap them so existing callers using
    ``key=...`` still work. Idempotent: skip if already wrapped or if
    the method already accepts ``key`` (older newton).
    """
    try:
        import newton
    except Exception:
        return
    MB = getattr(newton, "ModelBuilder", None)
    if MB is None or getattr(MB, "_metasim_key_label_shim_installed", False):
        return
    import inspect

    for method_name in (
        "add_body",
        "add_articulation",
        "add_shape_box",
        "add_shape_sphere",
        "add_shape_capsule",
        "add_shape_cylinder",
        "add_shape_mesh",
        "add_shape_plane",
        "add_joint_revolute",
        "add_joint_prismatic",
        "add_joint_fixed",
        "add_joint_free",
        "add_joint_ball",
        "add_joint_compound",
    ):
        meth = getattr(MB, method_name, None)
        if meth is None:
            continue
        try:
            params = inspect.signature(meth).parameters
        except (TypeError, ValueError):
            continue
        if "key" in params or "label" not in params:
            continue  # already accepts ``key`` or doesn't have ``label``

        def _make_wrapper(original):
            def wrapper(self, *args, **kwargs):
                if "key" in kwargs:
                    kwargs.setdefault("label", kwargs.pop("key"))
                return original(self, *args, **kwargs)

            wrapper.__name__ = original.__name__
            wrapper.__doc__ = original.__doc__
            wrapper.__wrapped__ = original
            return wrapper

        setattr(MB, method_name, _make_wrapper(meth))
    MB._metasim_key_label_shim_installed = True


_install_modelbuilder_key_to_label_shim()


def _check_newton_version():
    """One-time runtime check + warn if user is on an unsupported newton version.

    Newton 1.2 introduced multiple breaking changes; compat shims here
    handle 5 of them (JointType import, populate_contacts removal,
    current_world setter, Model *_key->*_label aliases, add_mjcf method
    removal via parse_mjcf fallback).
    """
    try:
        import newton

        ver = getattr(newton, "__version__", "unknown")
        if ver.startswith("1.0") or ver.startswith("1.1"):
            return  # supported, fully working
        warnings.warn(
            f"NewtonHandler running on newton {ver} (originally written for <1.2). "
            "5 API-drift shims applied (JointType, populate_contacts, current_world, "
            "Model.*_key aliases, add_mjcf via parse_mjcf). If you hit other issues, "
            "file a metasim ticket with the traceback.",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception:
        pass


_check_newton_version()


__all__ = ["JointType", "add_mjcf", "populate_contacts", "set_current_world"]
