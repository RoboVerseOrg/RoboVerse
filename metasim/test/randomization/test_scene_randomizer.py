# FIXME:
# This file may have many bugs, need to be reviewed and fixed.
"""Optimized tests for `SceneRandomizer`.

This rewrites the previous buggy test file to match the structure and
patterns used by other randomization tests (material/object/light).
Key fixes:
  - Use correct config field name `material_randomization` (was `material`).
  - Remove fragile string-based proxy calls; pass callables directly.
  - Add combined element test and env_ids scoping test.
  - Provide simple material pool paths pointing to existing local MDL assets.
  - Test sequential material selection deterministically via internal method.
  - Simplify standalone runner to reuse shared scenario helper.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.randomization.scene_randomizer import (
    SceneGeometryCfg,
    SceneMaterialPoolCfg,
    SceneRandomCfg,
    SceneRandomizer,
)
from metasim.test.randomization.conftest import get_shared_scenario


def _assert_contains(prims: list[str], substrings: list[str]):
    for s in substrings:
        assert any(s in p for p in prims), f"Expected prim containing '{s}' not found in created_prims: {prims}"


def scene_floor(handler, material_randomization: bool = False):
    cfg = SceneRandomCfg(
        floor=SceneGeometryCfg(
            enabled=True,
            size=(10.0, 10.0, 0.1),
            position=(0.0, 0.0, -0.05),
            material_randomization=material_randomization,
        ),
        floor_materials=(
            SceneMaterialPoolCfg(
                material_paths=[
                    "roboverse_data/materials/arnold/Wood/Ash.mdl",
                    "roboverse_data/materials/arnold/Wood/Birch.mdl",
                ],
                selection_strategy="random",
                randomize_material_variant=True,
            )
            if material_randomization
            else None
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=123)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    _assert_contains(props["created_prims"], ["scene_floor"])
    log.info(f"Scene floor test passed (material_randomization={material_randomization})")


def scene_walls(handler, material_randomization: bool = False):
    cfg = SceneRandomCfg(
        walls=SceneGeometryCfg(
            enabled=True,
            size=(6.0, 0.2, 3.0),
            position=(0.0, 0.0, 0.0),
            material_randomization=material_randomization,
        ),
        wall_materials=(
            SceneMaterialPoolCfg(
                material_paths=[
                    "roboverse_data/materials/arnold/Wood/Oak.mdl",
                    "roboverse_data/materials/arnold/Wood/Walnut.mdl",
                ],
                selection_strategy="random",
            )
            if material_randomization
            else None
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=456)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    # Walls exist per env; check one wall identifier substring
    _assert_contains(props["created_prims"], ["scene_wall_front"])
    log.info(f"Scene walls test passed (material_randomization={material_randomization})")


def scene_ceiling(handler, material_randomization: bool = False):
    cfg = SceneRandomCfg(
        ceiling=SceneGeometryCfg(
            enabled=True,
            size=(10.0, 10.0, 0.1),
            position=(0.0, 0.0, 3.0),
            material_randomization=material_randomization,
        ),
        ceiling_materials=(
            SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Wood/Cherry.mdl"],
                selection_strategy="random",
            )
            if material_randomization
            else None
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=789)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    _assert_contains(props["created_prims"], ["scene_ceiling"])
    log.info(f"Scene ceiling test passed (material_randomization={material_randomization})")


def scene_table(handler, material_randomization: bool = False):
    cfg = SceneRandomCfg(
        table=SceneGeometryCfg(
            enabled=True,
            size=(1.5, 1.0, 0.05),
            position=(0.5, 0.0, 0.4),
            material_randomization=material_randomization,
        ),
        table_materials=(
            SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Wood/Oak_Planks.mdl"],
                selection_strategy="random",
            )
            if material_randomization
            else None
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=1011)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    _assert_contains(props["created_prims"], ["scene_table"])
    log.info(f"Scene table test passed (material_randomization={material_randomization})")


def scene_combined(handler):
    cfg = SceneRandomCfg(
        floor=SceneGeometryCfg(
            enabled=True,
            size=(8.0, 8.0, 0.1),
            position=(0.0, 0.0, -0.05),
            material_randomization=False,
        ),
        walls=SceneGeometryCfg(
            enabled=True,
            size=(8.0, 0.2, 3.0),
            position=(0.0, 0.0, 0.0),
            material_randomization=False,
        ),
        ceiling=SceneGeometryCfg(
            enabled=True,
            size=(8.0, 8.0, 0.1),
            position=(0.0, 0.0, 3.0),
            material_randomization=False,
        ),
        table=SceneGeometryCfg(
            enabled=True,
            size=(1.5, 1.0, 0.05),
            position=(0.5, 0.0, 0.4),
            material_randomization=False,
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=2022)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    _assert_contains(
        props["created_prims"],
        ["scene_floor", "scene_wall_front", "scene_ceiling", "scene_table"],
    )
    log.info("Scene combined elements test passed")


def scene_material_selection(handler):
    # Test sequential selection determinism
    pool = SceneMaterialPoolCfg(
        material_paths=[
            "roboverse_data/materials/arnold/Wood/Ash.mdl",
            "roboverse_data/materials/arnold/Wood/Bamboo.mdl",
            "roboverse_data/materials/arnold/Wood/Cherry.mdl",
        ],
        selection_strategy="sequential",
        randomize_material_variant=False,
    )
    cfg = SceneRandomCfg(
        floor=SceneGeometryCfg(
            enabled=True,
            size=(2.0, 2.0, 0.1),
            position=(0.0, 0.0, -0.05),
            material_randomization=True,
        ),
        floor_materials=pool,
    )
    rand = SceneRandomizer(cfg, seed=999)
    rand.bind_handler(handler)
    first = rand._select_material(pool, "floor_index")
    second = rand._select_material(pool, "floor_index")
    third = rand._select_material(pool, "floor_index")
    assert first == pool.material_paths[0] and second == pool.material_paths[1] and third == pool.material_paths[2], (
        "Sequential material selection failed"
    )
    log.info("Scene material sequential selection test passed")


def scene_seed(handler):
    cfg = SceneRandomCfg(
        floor=SceneGeometryCfg(
            enabled=True,
            size=(10.0, 10.0, 0.1),
            position=(0.0, 0.0, -0.05),
            material_randomization=True,
        ),
        floor_materials=(
            SceneMaterialPoolCfg(
                material_paths=[
                    "roboverse_data/materials/arnold/Wood/Ash.mdl",
                    "roboverse_data/materials/arnold/Wood/Birch.mdl",
                ],
                selection_strategy="random",
                randomize_material_variant=True,
            )
        ),
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=555)
    rand.bind_handler(handler)
    rand.set_seed(42)
    rand()

    v1 = rand._rng.random()
    rand.set_seed(42)
    v2 = rand._rng.random()
    assert v1 == v2, "Same seed should reproduce RNG sequence"
    log.info("Scene seed reproducibility test passed")


def scene_env_ids(handler):
    # Randomize only walls for env 0
    cfg = SceneRandomCfg(
        walls=SceneGeometryCfg(
            enabled=True,
            size=(4.0, 0.2, 2.0),
            position=(0.0, 0.0, 0.0),
            material_randomization=False,
        ),
        env_ids=[0],
        only_if_no_scene=True,
    )
    rand = SceneRandomizer(cfg, seed=777)
    rand.bind_handler(handler)
    rand()
    props = rand.get_scene_properties()
    prims = props["created_prims"]
    assert any("env_0/scene_wall_front" in p for p in prims), "Env 0 wall should be created"
    # Ensure walls for other envs not created
    for env_id in range(1, handler.num_envs):
        assert not any(f"env_{env_id}/scene_wall_front" in p for p in prims), (
            f"Wall for env {env_id} should not be created"
        )
    log.info("Scene env_ids scoping test passed")


TEST_FUNCTIONS = [
    scene_floor,
    scene_walls,
    scene_ceiling,
    scene_table,
    scene_combined,
    scene_material_selection,
    scene_seed,
    scene_env_ids,
]


def _process_run_handler(scenario):
    from metasim.utils.setup_util import get_handler

    handler = get_handler(scenario)
    for func in TEST_FUNCTIONS:
        # Run material and non-material variants where applicable
        if func in (scene_floor, scene_walls, scene_ceiling, scene_table):
            func(handler, material_randomization=False)
            func(handler, material_randomization=True)
        else:
            func(handler)
    handler.close()


def run_test(sim: str = "isaacsim", num_envs: int = 2):
    if sim not in ["isaacsim"]:
        log.warning(f"Skipping: Only testing IsaacSim here, got {sim}")
        return
    scenario = get_shared_scenario(sim, num_envs)
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_process_run_handler, args=(scenario,))
    p.start()
    p.join(timeout=90)
    assert p.exitcode == 0, f"IsaacSim process exited abnormally: {p.exitcode}"
    log.info("Scene randomizer standalone test finished successfully")


@pytest.mark.usefixtures("shared_handler")
def test_scene_randomizer_with_shared_handler(shared_handler):
    log.info("Running scene randomizer tests with shared handler (proxy)")
    proxy = shared_handler
    for func in TEST_FUNCTIONS:
        if func in (scene_floor, scene_walls, scene_ceiling, scene_table):
            proxy.run_test(func, material_randomization=False)
            proxy.run_test(func, material_randomization=True)
        else:
            proxy.run_test(func)
    log.info("All scene randomizer tests completed with shared handler (proxy)")


if __name__ == "__main__":
    import sys

    sim = "isaacsim" if len(sys.argv) < 2 else sys.argv[1]
    num_envs = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    run_test(sim, num_envs)
