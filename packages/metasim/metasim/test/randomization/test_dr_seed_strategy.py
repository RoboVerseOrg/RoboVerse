"""DR-manager seed derivation is non-colliding and seed=0 safe (M1).

Three bugs in the old hand-picked-offset seeding scheme:

1. Loop bodies reused a single offset. The 4 wall material randomizers
   all received the same ``seed + 102`` → 4 walls were guaranteed to
   pick identical material on every run. Same shape for cameras.
2. ``+4+i`` for disk lights and ``+5+i`` for sphere lights collided
   (disk@i=1 == sphere@i=0 == seed+5). Two distinct randomizers shared
   a seed and therefore the same draw, which silently coupled the dome
   colour to one of the corner lights.
3. ``seed + 4 + i if seed else None`` (truthy check) treated ``seed=0``
   as "no seed", so callers passing 0 got a mix of seeded and unseeded
   randomizers.

After the fix, all randomizer seeds come from
``_derive_seed(base_seed, "<unique key>")``.
"""

from __future__ import annotations

import pytest

from metasim.randomization.dr_manager import _derive_seed

pytestmark = pytest.mark.general

# ---- Direct unit tests for the derivation helper ---------------------------


def test_derive_seed_none_in_none_out():
    assert _derive_seed(None, "anything") is None


def test_derive_seed_zero_is_a_valid_input():
    """``seed=0`` was the original truthy-check trap."""
    assert _derive_seed(0, "light.dome.0.dome") is not None
    assert isinstance(_derive_seed(0, "light.dome.0.dome"), int)


def test_derive_seed_is_deterministic_across_calls():
    a = _derive_seed(42, "material.wall.wall_front")
    b = _derive_seed(42, "material.wall.wall_front")
    assert a == b


def test_derive_seed_distinguishes_keys():
    """Different keys → different sub-seeds (probabilistically near-certain
    with 32 bits and only a handful of keys)."""
    s1 = _derive_seed(42, "material.wall.wall_front")
    s2 = _derive_seed(42, "material.wall.wall_back")
    s3 = _derive_seed(42, "material.wall.wall_left")
    s4 = _derive_seed(42, "material.wall.wall_right")
    assert len({s1, s2, s3, s4}) == 4


def test_derive_seed_distinguishes_base_seeds():
    """Same key, different base seeds → different sub-seeds."""
    a = _derive_seed(0, "camera.0.cam")
    b = _derive_seed(1, "camera.0.cam")
    c = _derive_seed(42, "camera.0.cam")
    assert len({a, b, c}) == 3


def test_derive_seed_fits_in_uint32():
    """Sub-seed fits in 32 bits; ``random.Random(seed)`` accepts any int but
    the rest of the code paths sometimes constrain to uint32."""
    for base in (0, 1, 42, 2**31, 2**63 - 1):
        sub = _derive_seed(base, "x")
        assert 0 <= sub < 2**32


# ---- Regression tests for the historic collisions -------------------------


def test_disk_and_sphere_lights_no_longer_collide():
    """Disk@i=1 used to share seed with Sphere@i=0 (both = base + 5)."""
    base = 7
    disk_0 = _derive_seed(base, "light.disk.0.disk_0")
    disk_1 = _derive_seed(base, "light.disk.1.disk_1")
    sph_0 = _derive_seed(base, "light.sphere.0.sphere_0")
    sph_1 = _derive_seed(base, "light.sphere.1.sphere_1")
    assert len({disk_0, disk_1, sph_0, sph_1}) == 4


def test_walls_no_longer_share_one_seed():
    """Pre-fix the entire wall loop reused ``wall_seed`` → identical mat
    on every wall. Now each wall has a unique sub-seed."""
    base = 42
    seeds = [_derive_seed(base, f"material.wall.{n}") for n in ("wall_front", "wall_back", "wall_left", "wall_right")]
    assert len(set(seeds)) == 4


def test_cameras_no_longer_share_one_seed():
    """Pre-fix every camera got ``seed + 10``. Now each camera index +
    name yields a distinct seed."""
    base = 42
    seeds = [_derive_seed(base, f"camera.{i}.cam_{i}") for i in range(3)]
    assert len(set(seeds)) == 3


def test_dome_light_when_base_seed_is_zero():
    """The truthy ``if seed else None`` bug turned ``seed=0`` into
    ``None`` only for dome lights. Now ``seed=0`` is treated like any
    other integer for every randomizer category."""
    assert _derive_seed(0, "light.disk.0.disk_a") is not None
    assert _derive_seed(0, "light.sphere.0.sphere_a") is not None
    assert _derive_seed(0, "light.dome.0.dome_a") is not None


# ---- Static-source check: the bad patterns are gone -----------------------


def test_no_remaining_hand_picked_seed_offsets():
    """``seed + N`` arithmetic for randomizer construction was the root of
    M1. Lock the regression: no source line should still do that.

    Allow the comment-only references in the helper docstring (lines
    that start with ``#`` or are inside a triple-quoted string), and
    allow ``SceneRandomizer(... seed=seed)`` (the un-offset base seed)
    by only matching ``seed + `` (followed by an int).
    """
    import re

    from metasim.randomization import dr_manager

    with open(dr_manager.__file__) as fh:
        text = fh.read()

    # Strip the module docstring and the ``_derive_seed`` docstring so
    # their explanatory references to "seed + 2" don't trip the check.
    no_docstrings = re.sub(r'"""[\s\S]*?"""', "", text)
    # Strip line comments.
    no_comments = re.sub(r"#.*", "", no_docstrings)

    bad = re.findall(r"\bseed\s*\+\s*\d+", no_comments)
    assert bad == [], f"hand-picked seed offsets still present: {bad}"


def test_no_truthy_seed_check():
    """Locked regression for the ``if seed else None`` bug."""
    import re

    from metasim.randomization import dr_manager

    with open(dr_manager.__file__) as fh:
        text = fh.read()
    no_docstrings = re.sub(r'"""[\s\S]*?"""', "", text)
    no_comments = re.sub(r"#.*", "", no_docstrings)

    # Forbid the truthy form ``if seed else``; require ``if seed is not None else``.
    bad = re.findall(r"if\s+seed\s+else", no_comments)
    assert bad == [], f"truthy seed check still present: {bad}"


def test_manager_batch_restores_flags_and_propagates_errors():
    from types import SimpleNamespace

    import pytest

    from metasim.randomization import DomainRandomizationManager, DRConfig

    events = []
    renderer = SimpleNamespace(
        _defer_all_visual_flushes=False,
        flush_visual_updates=lambda **kw: events.append("flush"),
        _invalidate_state_caches=lambda: events.append("invalidate"),
    )
    manager = DomainRandomizationManager.__new__(DomainRandomizationManager)
    manager.handler = SimpleNamespace(
        render_handler=renderer, _invalidate_state_caches=lambda: events.append("invalidate_hybrid")
    )
    manager.config = DRConfig(level=1)

    class Scene:
        cfg = SimpleNamespace(auto_flush_visuals=True)
        fail = True

        def __call__(self):
            assert renderer._defer_all_visual_flushes
            assert not self.cfg.auto_flush_visuals
            if self.fail:
                raise RuntimeError("scene failure")

    scene = Scene()
    manager.randomizers = {"scene": scene, "material_dynamic": []}
    with pytest.raises(RuntimeError, match="scene failure"):
        manager.apply_randomization()
    assert scene.cfg.auto_flush_visuals
    assert not renderer._defer_all_visual_flushes
    assert events == []
    scene.fail = False
    manager.apply_randomization()
    # Flush, then invalidate the renderer's and the wrapping hybrid's state caches.
    assert events == ["flush", "invalidate", "invalidate_hybrid"]
    renderer._defer_all_visual_flushes = True
    manager.apply_randomization()
    assert renderer._defer_all_visual_flushes
    # Inside an outer batch nothing flushes, but caches still go stale.
    assert events == ["flush", "invalidate", "invalidate_hybrid", "invalidate", "invalidate_hybrid"]
    renderer._defer_all_visual_flushes = False

    def failed_flush(**kwargs):
        raise RuntimeError("flush failure")

    renderer.flush_visual_updates = failed_flush
    with pytest.raises(RuntimeError, match="flush failure"):
        manager.apply_randomization()


def test_enabled_legacy_manager_rejects_non_isaac_renderer():
    from types import SimpleNamespace

    import pytest

    from metasim.randomization import DomainRandomizationManager, DRConfig

    handler = SimpleNamespace(cameras=[])
    with pytest.raises(NotImplementedError, match="VisualRandomizer"):
        DomainRandomizationManager(DRConfig(level=1), SimpleNamespace(), handler)
    # Level zero remains usable on every backend.
    manager = DomainRandomizationManager(DRConfig(), SimpleNamespace(), handler)
    assert manager.randomizers == {}


def test_legacy_manager_never_reseeds_global_rngs():
    """The manager derives per-randomizer seeds; it must not touch the caller's global generators.

    Level >= 1 needs a live Isaac Sim stage, so the guard is on the source: no global
    seeding call may return to dr_manager (the removed ``_setup_reproducibility``).
    """
    import inspect
    import re

    from metasim.randomization import dr_manager

    source = inspect.getsource(dr_manager)
    forbidden = re.findall(r"\b(random\.seed|np\.random\.seed|numpy\.random\.seed|torch\.manual_seed)\s*\(", source)
    assert forbidden == [], f"global RNG reseeding returned to dr_manager: {forbidden}"
    assert "_setup_reproducibility" not in source


def test_legacy_config_rejects_invalid_levels_and_modes():
    import pytest

    from metasim.randomization import DRConfig

    for kwargs in ({"level": 4}, {"level": True}, {"scene_mode": -1}, {"scene_mode": 1.5}):
        with pytest.raises(ValueError):
            DRConfig(**kwargs)
