from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg
from roboverse_pack.benchmark import get_benchmark_task_spec
from roboverse_pack.tasks.benchmark.base import build_benchmark_scenario

INDOOR_SCENE_OBJECTS = {
    "indoor_floor",
    "indoor_back_wall",
    "indoor_left_wall",
    "indoor_right_wall",
    "walnut_console",
    "cream_area_rug",
    "brass_wall_trim",
    "marble_display_plinth",
}

BEDROOM_SCENE_OBJECTS = {
    "bed_frame",
    "bed_headboard",
    "bed_mattress",
    "bed_duvet",
    "bed_pillow_left",
    "bed_pillow_right",
    "left_nightstand",
    "right_nightstand",
    "bedside_lamp_base",
    "bedside_lamp_shade",
    "wardrobe",
    "window_glow_panel",
    "window_frame_top",
    "window_frame_bottom",
    "window_frame_left",
    "window_frame_right",
}


def _build_scenario(simulator: str):
    return build_benchmark_scenario(
        get_benchmark_task_spec("benchmark.cube_reach"),
        robot="openarm_bimanual_wuji",
        simulator=simulator,
    )


def _scenario_object_map(simulator: str):
    scenario = _build_scenario(simulator)
    return {obj.name: obj for obj in scenario.objects}


def test_blender_glass_example_includes_elegant_indoor_scene() -> None:
    objects = _scenario_object_map("blender")

    assert INDOOR_SCENE_OBJECTS | BEDROOM_SCENE_OBJECTS <= set(objects)
    for name in INDOOR_SCENE_OBJECTS | BEDROOM_SCENE_OBJECTS:
        assert isinstance(objects[name], PrimitiveCubeCfg)
        assert objects[name].physics == PhysicStateType.XFORM

    assert objects["indoor_floor"].size[0] > 1.5
    assert objects["indoor_floor"].size[1] > 1.4
    assert objects["indoor_back_wall"].default_position[1] > -0.05


def test_mujoco_physics_scene_matches_blender_static_indoor_props() -> None:
    blender_objects = _scenario_object_map("blender")
    mujoco_objects = _scenario_object_map("mujoco")

    assert INDOOR_SCENE_OBJECTS | BEDROOM_SCENE_OBJECTS <= set(mujoco_objects)
    for name in INDOOR_SCENE_OBJECTS | BEDROOM_SCENE_OBJECTS:
        assert mujoco_objects[name].default_position == blender_objects[name].default_position
        assert mujoco_objects[name].default_orientation == blender_objects[name].default_orientation
        assert mujoco_objects[name].size == blender_objects[name].size


def test_isaacsim_benchmark_scene_does_not_add_blender_indoor_props() -> None:
    objects = _scenario_object_map("isaacsim")

    assert set(objects).isdisjoint(INDOOR_SCENE_OBJECTS)


def test_blender_glass_example_uses_warm_indoor_lighting() -> None:
    lights = {light.name: light for light in _build_scenario("blender").lights}

    ambient = lights["indoor_warm_ambient"]
    ceiling_key = lights["indoor_ceiling_key"]
    lamp_accent = lights["desk_lamp_warm_accent"]

    assert ambient.color[0] - ambient.color[2] > 0.35
    assert isinstance(ceiling_key, DiskLightCfg)
    assert ceiling_key.color[0] - ceiling_key.color[2] > 0.45
    assert 150.0 <= ceiling_key.intensity <= 600.0
    assert isinstance(lamp_accent, SphereLightCfg)
    assert lamp_accent.color[0] - lamp_accent.color[2] > 0.65
    assert lamp_accent.pos[:2] == (-0.43, -0.82)


def test_isaacsim_benchmark_scene_keeps_default_lighting() -> None:
    lights = {light.name: light for light in _build_scenario("isaacsim").lights}

    assert "indoor_ceiling_key" not in lights
    assert "desk_lamp_warm_accent" not in lights


def test_overview_camera_stays_inside_indoor_side_walls() -> None:
    scenario = _build_scenario("blender")
    objects = {obj.name: obj for obj in scenario.objects}
    overview_camera = next(camera for camera in scenario.cameras if camera.name == "overview")

    left_wall = objects["indoor_left_wall"]
    right_wall = objects["indoor_right_wall"]
    left_inner_face_x = left_wall.default_position[0] + left_wall.size[0] / 2.0
    right_inner_face_x = right_wall.default_position[0] - right_wall.size[0] / 2.0

    assert left_inner_face_x < overview_camera.pos[0] < right_inner_face_x
    assert right_inner_face_x - overview_camera.pos[0] > 0.20


def test_bedroom_furniture_stays_behind_tabletop_work_area() -> None:
    objects = _scenario_object_map("blender")
    table = objects["table"]

    for name in ("bed_frame", "bed_mattress", "bed_duvet", "left_nightstand", "right_nightstand"):
        assert objects[name].default_position[1] > table.default_position[1] + table.size[1] / 2.0
