"""The MuJoCo handler must reserve a large enough arena for humanoid + mesh scenes.

MuJoCo >= 3 sizes its arena from ``<size memory>``; the compiler default made
``get_started/10_mount_camera.py`` (H1 + objects) die with ``mj_stackAlloc: out of memory``.
"""

from __future__ import annotations

import pytest


@pytest.mark.mujoco
def test_arena_memory_is_reserved(handler):
    """The compiled model carries at least the handler's default arena (512 MB)."""
    from metasim.sim.mujoco.mujoco import MUJOCO_ARENA_MEMORY

    assert MUJOCO_ARENA_MEMORY.endswith("M")
    expected_bytes = int(MUJOCO_ARENA_MEMORY[:-1]) * 2**20
    narena = int(handler.physics.model.narena)
    assert narena >= expected_bytes, f"model.narena={narena} < {expected_bytes}: <size memory> was not applied"
