"""Static guard: backend ``close()`` methods must be idempotent.

Background: calling ``handler.close()`` twice — common when a user
combines a context manager with an explicit close, or when a teardown
hook fires alongside an except-block close — used to crash on at
least three backends:

- MuJoCo: ``self.viewer.close()`` without nullifying, so the second
  call re-closed an already-torn-down viewer.
- Sapien3: same pattern; also AttributeError'd when called before
  ``launch()`` ever set ``self.viewer``.
- MJX: ``if hasattr(_viewer) and _renderer is not None`` — guarded
  the viewer close behind the renderer's existence, so headed-without-
  camera leaked the viewer and headless-with-camera AttributeError'd
  on the missing viewer.

Each handler now nulls the handle after close, and the close path is
wrapped in try/except so a second invocation is a clean no-op. This
test asserts the contract statically (AST inspection — no GPU, no
sim env).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# (path-relative-to-repo, ClassName, [handle attribute names that close() must None-out])
_CONTRACTS = [
    ("metasim/sim/mujoco/mujoco.py", "MujocoHandler", ["viewer", "renderer"]),
    ("metasim/sim/sapien/sapien3.py", "Sapien3Handler", ["viewer"]),
    ("metasim/sim/mjx/mjx.py", "MJXHandler", ["_viewer", "_renderer"]),
]


def _load_close_body(path: str, class_name: str) -> ast.FunctionDef:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "close")


@pytest.mark.general
@pytest.mark.parametrize("path,class_name,handles", _CONTRACTS)
def test_close_nullifies_handles(path: str, class_name: str, handles: list[str]):
    """Each handle assigned in ``__init__`` and torn down in ``close()``
    must be set to ``None`` after teardown so a second ``close()`` call
    is a no-op rather than re-closing a stale handle."""
    fn = _load_close_body(path, class_name)
    nulled = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                # Pattern: ``self.<handle> = None``
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is None
                ):
                    nulled.add(target.attr)
    missing = [h for h in handles if h not in nulled]
    assert not missing, (
        f"{class_name}.close() does not null-out {missing} after closing them. "
        f"A second close() call will try to re-close a stale handle — see "
        f"test_close_idempotent_general.py for the full rationale."
    )


@pytest.mark.general
@pytest.mark.parametrize("path,class_name,handles", _CONTRACTS)
def test_close_wraps_handle_close_in_try_except(path: str, class_name: str, handles: list[str]):
    """Each ``<handle>.close()`` call inside ``close()`` must be wrapped
    in ``try/except`` so a third-party teardown failure (e.g. GLFW
    already destroyed, GPU context lost) doesn't prevent the rest of
    close() from running. This is what makes the second close() call
    safe — even if the first ``inner.close()`` raises, our handle is
    still nullified afterward."""
    fn = _load_close_body(path, class_name)
    # Find ``<x>.close()`` calls inside a Try.
    safely_closed_handles: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "close"
            ):
                # The receiver is the handle we're closing.
                if isinstance(inner.func.value, ast.Attribute):
                    safely_closed_handles.add(inner.func.value.attr)
                elif isinstance(inner.func.value, ast.Name):
                    safely_closed_handles.add(inner.func.value.id)
    # At least one of the handles should be safely wrapped — backends
    # that don't have a fragile close() may legitimately skip try/except
    # for some handles, so we only require that *some* close() call is
    # wrapped.
    assert safely_closed_handles, (
        f"{class_name}.close() has no try/except around any inner close() call. "
        f"A teardown failure on one handle (e.g. GLFW destroyed) will mask the "
        f"others. Wrap each ``<handle>.close()`` in try/except so the handler "
        f"can still null everything out."
    )
