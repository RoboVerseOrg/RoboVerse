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
