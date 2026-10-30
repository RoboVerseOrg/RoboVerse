"""``metasim.kernel`` is the minimum a benchmark port builds on: backend-free and API-pinned."""

from __future__ import annotations

import subprocess
import sys

import pytest

_BACKENDS = ("mujoco", "mujoco_warp", "newton", "warp", "isaacgym", "isaacsim", "omni", "sapien", "pybullet", "genesis")


@pytest.mark.general
def test_importing_the_kernel_loads_no_simulator_backend():
    code = (
        "import sys, metasim.kernel\n"
        f"loaded = sorted({{m.split('.')[0] for m in sys.modules}} & set({_BACKENDS!r}))\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"metasim.kernel imports simulator backends: {out}"


@pytest.mark.general
def test_every_kernel_name_is_pinned_by_the_public_api_snapshot():
    """A kernel name is a promise; its defining module must be in the snapshot so a break fails CI."""
    import metasim.kernel as kernel
    from metasim.utils.api_surface import PUBLIC_MODULES

    unpinned = []
    for name in kernel.__all__:
        obj = getattr(kernel, name)
        module = getattr(obj, "__module__", None)
        if module not in PUBLIC_MODULES:
            unpinned.append(f"{name} ({module})")
    assert not unpinned, f"kernel names defined outside PUBLIC_MODULES: {unpinned}"
