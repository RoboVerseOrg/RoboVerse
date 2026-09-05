"""Close a handler's ``mujoco.Renderer`` when the handler goes away and before MuJoCo tears down EGL.

MuJoCo registers ``eglTerminate`` with ``atexit`` when the display is created (inside the first
``Renderer``). A renderer still alive at interpreter exit then frees its context after the display
is gone and ``Renderer.__del__`` ends the process in an ``EGLError`` traceback. ``weakref.finalize``
alone does not help: its ``atexit`` hook dates from the first finalizer any import created, which is
earlier, so it runs after the display is terminated. An ``atexit`` hook registered right after the
first renderer is created runs before ``eglTerminate`` (LIFO); one hook per handler, holding a weak
reference, that runs every renderer finalizer the handler was given, so a renderer that was replaced
(the macOS path does so per camera size) or dropped without ``close()`` is closed in time too.

Shared by the MuJoCo and MJX handlers, which each own one offscreen renderer at a time.
"""

from __future__ import annotations

import atexit
import weakref


def close_renderer(renderer) -> None:
    """Close ``renderer``; idempotent, and quiet when the GL context is already gone at exit."""
    try:
        renderer.close()
    except Exception:  # nothing left to release once the display is terminated
        pass


def _run_finalizers(owner_ref) -> None:
    owner = owner_ref()
    for finalizer in list(getattr(owner, "_renderer_finalizers", ()) if owner is not None else ()):
        finalizer()  # a finalizer that already ran is a no-op


def attach_renderer_teardown(owner, renderer) -> weakref.finalize:
    """Register the teardown of ``renderer`` for ``owner``; returns the finalizer.

    Call the finalizer from ``close()`` or when replacing the renderer: it closes this renderer once.
    The renderer is also closed when ``owner`` is garbage-collected, and at interpreter exit before
    MuJoCo's display is terminated, whether or not the handler still points at it.
    """
    finalizer = weakref.finalize(owner, close_renderer, renderer)
    finalizers = getattr(owner, "_renderer_finalizers", None)
    if finalizers is None:
        finalizers = owner._renderer_finalizers = []
        atexit.register(_run_finalizers, weakref.ref(owner))
    finalizers[:] = [f for f in finalizers if f.alive] + [finalizer]
    return finalizer
