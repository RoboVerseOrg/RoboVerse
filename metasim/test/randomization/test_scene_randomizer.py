from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.scene_randomizer import (
    ManualGeometryCfg,
    SceneLayerCfg,
    SceneRandomCfg,
    SceneRandomizer,
    USDAssetCfg,
    USDAssetPoolCfg,
)


def scene_floor(handler):
    """Test creating a floor using environment layer."""
    cfg = SceneRandomCfg(
        environment_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="floor",
                    geometry_type="cube",
                    size=(10.0, 10.0, 0.1),
                    position=(0.0, 0.0, -0.05),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=123)
    rand.bind_handler(handler)
    rand()
    # Check that floor prim was created
    assert len(rand._created_prims) > 0, "Expected floor prim to be created"
    assert "floor" in rand._created_prims, "Expected 'floor' in created_prims"
    log.info("Scene floor test passed")


def scene_walls(handler):
    """Test creating walls using environment layer."""
    cfg = SceneRandomCfg(
        environment_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="wall_front",
                    geometry_type="cube",
                    size=(6.0, 0.2, 3.0),
                    position=(3.0, 0.0, 1.5),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=456)
    rand.bind_handler(handler)
    rand()
    assert len(rand._created_prims) > 0, "Expected wall prim to be created"
    assert "wall_front" in rand._created_prims, "Expected 'wall_front' in created_prims"
    log.info("Scene walls test passed")


def scene_ceiling(handler):
    """Test creating a ceiling using environment layer."""
    cfg = SceneRandomCfg(
        environment_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="ceiling",
                    geometry_type="cube",
                    size=(10.0, 10.0, 0.1),
                    position=(0.0, 0.0, 3.0),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=789)
    rand.bind_handler(handler)
    rand()
    assert len(rand._created_prims) > 0, "Expected ceiling prim to be created"
    assert "ceiling" in rand._created_prims, "Expected 'ceiling' in created_prims"
    log.info("Scene ceiling test passed")


def scene_table(handler):
    """Test creating a table using workspace layer."""
    cfg = SceneRandomCfg(
        workspace_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.5, 1.0, 0.05),
                    position=(0.5, 0.0, 0.4),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=1011)
    rand.bind_handler(handler)
    rand()
    assert len(rand._created_prims) > 0, "Expected table prim to be created"
    assert "table" in rand._created_prims, "Expected 'table' in created_prims"
    log.info("Scene table test passed")


def scene_combined(handler):
    """Test combining multiple layers (environment + workspace)."""
    cfg = SceneRandomCfg(
        environment_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="floor",
                    geometry_type="cube",
                    size=(8.0, 8.0, 0.1),
                    position=(0.0, 0.0, -0.05),
                    enabled=True,
                ),
                ManualGeometryCfg(
                    name="wall_front",
                    geometry_type="cube",
                    size=(8.0, 0.2, 3.0),
                    position=(4.0, 0.0, 1.5),
                    enabled=True,
                ),
                ManualGeometryCfg(
                    name="ceiling",
                    geometry_type="cube",
                    size=(8.0, 8.0, 0.1),
                    position=(0.0, 0.0, 3.0),
                    enabled=True,
                ),
            ],
        ),
        workspace_layer=SceneLayerCfg(
            shared=True,
            z_offset=0.0,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.5, 1.0, 0.05),
                    position=(0.5, 0.0, 0.4),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=2022)
    rand.bind_handler(handler)
    rand()
    assert len(rand._created_prims) > 0, "Expected prims to be created"
    expected_names = ["floor", "wall_front", "ceiling", "table"]
    for name in expected_names:
        assert name in rand._created_prims, f"Expected '{name}' in created_prims"
    log.info("Scene combined elements test passed")


def scene_usd_asset_pool(handler):
    """Test USD asset pool sequential selection."""
    # Test USDAssetPoolCfg with sequential selection
    pool = USDAssetPoolCfg(
        name="test_pool",
        usd_paths=[
            "roboverse_data/assets/test1.usd",
            "roboverse_data/assets/test2.usd",
            "roboverse_data/assets/test3.usd",
        ],
        selection_strategy="sequential",
        position=(0.0, 0.0, 0.5),
    )

    cfg = SceneRandomCfg(
        workspace_layer=SceneLayerCfg(
            shared=True,
            enabled=True,
            elements=[pool],
        ),
    )
    rand = SceneRandomizer(cfg, seed=999)
    rand.bind_handler(handler)

    # Test that pool candidates were created correctly
    assert pool.candidates is not None, "Pool candidates should be auto-generated"
    assert len(pool.candidates) == 3, "Expected 3 candidates"
    log.info("Scene USD asset pool test passed")


def scene_seed(handler):
    """Test seed reproducibility."""
    cfg = SceneRandomCfg(
        environment_layer=SceneLayerCfg(
            shared=True,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="floor",
                    geometry_type="cube",
                    size=(10.0, 10.0, 0.1),
                    position=(0.0, 0.0, -0.05),
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=555)
    rand.bind_handler(handler)
    rand.set_seed(42)

    v1 = rand._rng.random()
    rand.set_seed(42)
    v2 = rand._rng.random()
    assert v1 == v2, "Same seed should reproduce RNG sequence"
    log.info("Scene seed reproducibility test passed")


def scene_default_material(handler):
    """Test ManualGeometryCfg with default_material."""
    cfg = SceneRandomCfg(
        workspace_layer=SceneLayerCfg(
            shared=True,
            enabled=True,
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.5, 1.0, 0.05),
                    position=(0.5, 0.0, 0.4),
                    default_material="roboverse_data/materials/arnold/Wood/Oak.mdl",
                    enabled=True,
                )
            ],
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=777)
    rand.bind_handler(handler)
    rand()
    assert "table" in rand._created_prims, "Expected 'table' in created_prims"
    log.info("Scene default material test passed")


_SCENE_CASES = [
    pytest.param(scene_floor, {}, id="scene_floor"),
    pytest.param(scene_walls, {}, id="scene_walls"),
    pytest.param(scene_ceiling, {}, id="scene_ceiling"),
    pytest.param(scene_table, {}, id="scene_table"),
    pytest.param(scene_combined, {}, id="scene_combined"),
    pytest.param(scene_usd_asset_pool, {}, id="scene_usd_asset_pool"),
    pytest.param(scene_seed, {}, id="scene_seed"),
    pytest.param(scene_default_material, {}, id="scene_default_material"),
]


@pytest.mark.isaacsim
@pytest.mark.parametrize(("test_func", "kwargs"), _SCENE_CASES)
def test_scene_randomizers(handler, test_func, kwargs):
    """Run scene randomizer checks inside the shared handler process."""
    test_func(handler, **kwargs)


# -----------------------------------------------------------------------------
# General (no-simulator) tests: pool candidate expansion + placement transform
# -----------------------------------------------------------------------------


@pytest.mark.general
def test_usd_pool_explicit_candidates_bypass_usd_paths():
    """Explicit ``candidates`` are accepted verbatim, without ``usd_paths``.

    The ``usd_paths`` expansion keys ``per_path_overrides`` by USD path, so it cannot express
    the same USD at several DISTINCT poses; an explicit candidate list is the supported way
    (e.g. one apartment USD placed at multiple workcell placements).
    """
    same_usd = "roboverse_data/assets/apartment.usd"
    pool = USDAssetPoolCfg(
        name="apartment_placements",
        candidates=[
            USDAssetCfg(name="pose0", usd_path=same_usd, position=(0.0, 0.0, 0.0)),
            USDAssetCfg(name="pose1", usd_path=same_usd, position=(2.0, -1.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        ],
    )
    assert [c.name for c in pool.candidates] == ["pose0", "pose1"]
    assert [c.position for c in pool.candidates] == [(0.0, 0.0, 0.0), (2.0, -1.0, 0.0)]
    assert pool.candidates[1].rotation == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.general
def test_usd_pool_explicit_candidates_win_over_usd_paths():
    """When both are given, ``candidates`` win and ``usd_paths`` is ignored (documented contract)."""
    pool = USDAssetPoolCfg(
        name="both",
        usd_paths=["a.usd", "b.usd"],
        candidates=[USDAssetCfg(name="only", usd_path="c.usd")],
    )
    assert [c.usd_path for c in pool.candidates] == ["c.usd"]


@pytest.mark.general
def test_usd_pool_requires_usd_paths_or_candidates():
    """A pool with neither ``usd_paths`` nor ``candidates`` fails fast."""
    with pytest.raises(ValueError, match="provide usd_paths or candidates"):
        USDAssetPoolCfg(name="empty")


@pytest.mark.general
def test_usd_pool_rejects_empty_explicit_candidates():
    """An explicit empty ``candidates`` list is an error, never a silent fallback."""
    with pytest.raises(ValueError, match="explicit candidates cannot be empty"):
        USDAssetPoolCfg(name="empty_explicit", candidates=[])
    # ... even when usd_paths is also given (no silent expansion of usd_paths)
    with pytest.raises(ValueError, match="explicit candidates cannot be empty"):
        USDAssetPoolCfg(name="empty_explicit_with_paths", usd_paths=["a.usd"], candidates=[])


def _offline_scene_randomizer():
    """SceneRandomizer on a plain in-memory USD stage, no simulator.

    ``__init__`` imports IsaacSim prim utils, so bypass it and provide only the state the
    placement-transform paths (``_apply_transform`` / same-USD ``_load_or_replace_usd``) use.
    """
    pytest.importorskip("pxr", reason="needs usd-core (pxr) for stage-level checks")
    from pxr import Usd

    rand = SceneRandomizer.__new__(SceneRandomizer)
    rand.stage = Usd.Stage.CreateInMemory()
    rand._loaded_usds = {}
    return rand


@pytest.mark.general
def test_apply_transform_authors_translate_orient_scale_stack():
    """_apply_transform authors a deterministic [translate, orient, scale] stack.

    Translate is OUTERMOST (candidate ``position`` is the pool-convention ``dp``), z_offset is
    added to z, and orient is ALWAYS written -- as float precision (Quatf), matching the
    handler's scene-load path -- even when the placement rotation is identity.
    """
    pytest.importorskip("pxr", reason="needs usd-core (pxr) for stage-level checks")
    from pxr import Gf, UsdGeom

    rand = _offline_scene_randomizer()
    UsdGeom.Xform.Define(rand.stage, "/scene_pool")
    element = USDAssetCfg(
        name="apartment",
        usd_path="apartment.usd",
        position=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=(2.0, 2.0, 2.0),
    )
    rand._apply_transform("/scene_pool", element, z_offset=0.5)

    ops = UsdGeom.Xformable(rand.stage.GetPrimAtPath("/scene_pool")).GetOrderedXformOps()
    assert [op.GetOpName() for op in ops] == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert ops[0].Get() == Gf.Vec3d(1.0, 2.0, 3.5)
    assert ops[1].Get() == Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))
    assert ops[2].Get() == Gf.Vec3d(2.0, 2.0, 2.0)


@pytest.mark.general
def test_apply_transform_overrides_preexisting_op_stack():
    """A referenced asset root with its own op stack still gets the full placement.

    Regression: the old in-place update path only wrote orient when the asset already had an
    orient op AND the rotation was non-identity, silently dropping the placement yaw for asset
    roots authored with e.g. a translate-only stack.
    """
    pytest.importorskip("pxr", reason="needs usd-core (pxr) for stage-level checks")
    from pxr import Gf, UsdGeom

    rand = _offline_scene_randomizer()
    xf = UsdGeom.Xform.Define(rand.stage, "/scene_pool")
    xf.AddTranslateOp().Set(Gf.Vec3d(5.0, 5.0, 5.0))  # asset-authored stack without orient

    element = USDAssetCfg(
        name="apartment",
        usd_path="apartment.usd",
        position=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),  # 180 deg yaw
    )
    rand._apply_transform("/scene_pool", element, z_offset=0.0)

    ops = UsdGeom.Xformable(rand.stage.GetPrimAtPath("/scene_pool")).GetOrderedXformOps()
    assert [op.GetOpName() for op in ops] == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert ops[0].Get() == Gf.Vec3d(1.0, 2.0, 3.0)
    assert ops[1].Get() == Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))


@pytest.mark.general
def test_apply_transform_matches_preexisting_op_precision():
    """A surviving double-precision orient attribute is re-authored, not crashed on.

    ``ClearXformOpOrder`` resets only the op ORDER; op attributes composed in from the asset
    survive at their original precision, and ``AddOrientOp``'s default float precision raises
    on a surviving ``quatd``. _apply_transform must match the existing precision instead.
    """
    pytest.importorskip("pxr", reason="needs usd-core (pxr) for stage-level checks")
    from pxr import Gf, UsdGeom

    rand = _offline_scene_randomizer()
    xf = UsdGeom.Xform.Define(rand.stage, "/scene_pool")
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    element = USDAssetCfg(
        name="apartment",
        usd_path="apartment.usd",
        position=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    rand._apply_transform("/scene_pool", element, z_offset=0.0)

    ops = UsdGeom.Xformable(rand.stage.GetPrimAtPath("/scene_pool")).GetOrderedXformOps()
    assert [op.GetOpName() for op in ops] == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert ops[1].GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    assert ops[1].Get() == Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, 1.0))


@pytest.mark.general
def test_same_usd_reselection_reapplies_placement():
    """Re-selecting the SAME USD with a different candidate pose moves the prim.

    Regression: ``_load_or_replace_usd`` used to skip when the USD path was unchanged, leaving
    the previous placement -- one apartment USD placed at several workcell poses never moved,
    and a stale rotation survived re-selection.
    """
    pytest.importorskip("pxr", reason="needs usd-core (pxr) for stage-level checks")
    from pxr import Gf, UsdGeom

    rand = _offline_scene_randomizer()
    UsdGeom.Xform.Define(rand.stage, "/scene_pool")
    first = USDAssetCfg(name="pose0", usd_path="apartment.usd", position=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.0, 1.0))
    rand._apply_transform("/scene_pool", first, z_offset=0.0)
    rand._loaded_usds["/scene_pool"] = first.usd_path  # as if _load_usd had run

    second = USDAssetCfg(name="pose1", usd_path="apartment.usd", position=(9.0, 8.0, 7.0))
    rand._load_or_replace_usd("/scene_pool", second, z_offset=0.0)

    ops = UsdGeom.Xformable(rand.stage.GetPrimAtPath("/scene_pool")).GetOrderedXformOps()
    assert ops[0].Get() == Gf.Vec3d(9.0, 8.0, 7.0)
    # pose0's yaw must not survive; pose1's identity rotation is re-authored
    assert ops[1].Get() == Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
