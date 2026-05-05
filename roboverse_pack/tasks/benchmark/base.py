from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import BaseLightCfg, DiskLightCfg, DomeLightCfg, SphereLightCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_handler, get_robot
from roboverse_pack.benchmark import BenchmarkObjectSpec, BenchmarkTaskSpec
from roboverse_pack.teleop.flow import ControlCommand, StepResult

TABLE_POS_CM = (0.0, -55.0, 18.0)
TABLE_SIZE_CM = (120.0, 90.0, 4.0)
TABLE_COLOR_RGB = [0.33, 0.24, 0.18]
BENCHMARK_ASSET_DIR = Path(__file__).resolve().parent / "assets"
BOTTLE_ASSET_DIR = BENCHMARK_ASSET_DIR / "elegant_glass_bottle"
BOTTLE_POS_M = (0.24, -0.42, 0.20)
CRYSTAL_TUMBLER_POS_M = (-0.43, -0.34, 0.20)
GLASS_HURRICANE_POS_M = (0.45, -0.31, 0.20)
DISPLAY_CASE_POS_M = (0.40, -0.78, 0.20)
DESK_LAMP_POS_M = (-0.43, -0.82, 0.20)
INDOOR_SCENE_SIMULATORS = {"mujoco", "blender"}


def _uses_indoor_scene(task_spec: BenchmarkTaskSpec, simulator: str) -> bool:
    return task_spec.name == "benchmark.cube_reach" and simulator in INDOOR_SCENE_SIMULATORS


def _decor_cube_cfg(
    name: str,
    *,
    size: tuple[float, float, float],
    color: list[float],
    position: tuple[float, float, float],
) -> PrimitiveCubeCfg:
    return PrimitiveCubeCfg(
        name=name,
        size=list(size),
        mass=1.0,
        physics=PhysicStateType.XFORM,
        color=color,
        default_position=position,
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _cm_vec_to_m(values_cm: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(values_cm[0]) / 100.0, float(values_cm[1]) / 100.0, float(values_cm[2]) / 100.0)


def _object_size_m(obj: BenchmarkObjectSpec) -> list[float]:
    return [2.0 * float(obj.half_extents_cm[index]) / 100.0 for index in range(3)]


def _primitive_cube_cfg(obj: BenchmarkObjectSpec) -> PrimitiveCubeCfg:
    return PrimitiveCubeCfg(
        name=obj.name,
        size=_object_size_m(obj),
        mass=0.02 if obj.movable else 1.0,
        physics=PhysicStateType.RIGIDBODY if obj.movable else PhysicStateType.XFORM,
        color=[0.82, 0.26, 0.18] if obj.movable else [0.7, 0.7, 0.7],
        default_position=_cm_vec_to_m(obj.pos_cm),
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _table_cfg() -> PrimitiveCubeCfg:
    return PrimitiveCubeCfg(
        name="table",
        size=list(_cm_vec_to_m(TABLE_SIZE_CM)),
        mass=1.0,
        physics=PhysicStateType.GEOM,
        color=list(TABLE_COLOR_RGB),
        default_position=_cm_vec_to_m(TABLE_POS_CM),
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _elegant_glass_bottle_cfg() -> RigidObjCfg:
    file_type = dict(RigidObjCfg.file_type)
    file_type["blender"] = "mesh"
    return RigidObjCfg(
        name="elegant_glass_bottle",
        physics=PhysicStateType.RIGIDBODY,
        mjcf_path=str(BOTTLE_ASSET_DIR / "elegant_glass_bottle.xml"),
        mesh_path=str(BOTTLE_ASSET_DIR / "elegant_glass_bottle.fbx"),
        file_type=file_type,
        default_position=BOTTLE_POS_M,
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _mjcf_blender_file_type(base_file_type: dict[str, str]) -> dict[str, str]:
    file_type = dict(base_file_type)
    file_type["blender"] = "mjcf"
    return file_type


def _rigid_mjcf_prop_cfg(name: str, default_position: tuple[float, float, float]) -> RigidObjCfg:
    return RigidObjCfg(
        name=name,
        physics=PhysicStateType.RIGIDBODY,
        mjcf_path=str(BENCHMARK_ASSET_DIR / name / f"{name}.xml"),
        file_type=_mjcf_blender_file_type(RigidObjCfg.file_type),
        default_position=default_position,
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _articulated_mjcf_prop_cfg(name: str, default_position: tuple[float, float, float]) -> ArticulationObjCfg:
    return ArticulationObjCfg(
        name=name,
        mjcf_path=str(BENCHMARK_ASSET_DIR / name / f"{name}.xml"),
        file_type=_mjcf_blender_file_type(ArticulationObjCfg.file_type),
        fix_base_link=True,
        default_position=default_position,
        default_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def _indoor_scene_cfgs() -> list[PrimitiveCubeCfg]:
    return [
        _decor_cube_cfg(
            "indoor_floor",
            size=(3.50, 1.90, 0.05),
            color=[0.52, 0.49, 0.44],
            position=(0.0, -0.50, -0.025),
        ),
        _decor_cube_cfg(
            "cream_area_rug",
            size=(1.55, 1.12, 0.012),
            color=[0.86, 0.80, 0.68],
            position=(0.0, -0.58, 0.006),
        ),
        _decor_cube_cfg(
            "indoor_back_wall",
            size=(3.50, 0.045, 1.40),
            color=[0.84, 0.81, 0.74],
            position=(0.0, 0.18, 0.70),
        ),
        _decor_cube_cfg(
            "indoor_left_wall",
            size=(0.045, 1.90, 1.40),
            color=[0.74, 0.73, 0.68],
            position=(-1.65, -0.50, 0.70),
        ),
        _decor_cube_cfg(
            "indoor_right_wall",
            size=(0.045, 1.90, 1.40),
            color=[0.78, 0.76, 0.70],
            position=(1.65, -0.50, 0.70),
        ),
        _decor_cube_cfg(
            "walnut_console",
            size=(1.36, 0.18, 0.12),
            color=[0.24, 0.15, 0.09],
            position=(0.0, 0.03, 0.24),
        ),
        _decor_cube_cfg(
            "brass_wall_trim",
            size=(1.86, 0.018, 0.026),
            color=[0.76, 0.55, 0.25],
            position=(0.0, 0.153, 0.57),
        ),
        _decor_cube_cfg(
            "marble_display_plinth",
            size=(0.26, 0.22, 0.08),
            color=[0.78, 0.76, 0.71],
            position=(-0.43, -0.34, 0.14),
        ),
        *_bedroom_scene_cfgs(),
    ]


def _bedroom_scene_cfgs() -> list[PrimitiveCubeCfg]:
    return [
        _decor_cube_cfg(
            "bed_frame",
            size=(1.05, 0.54, 0.10),
            color=[0.25, 0.13, 0.07],
            position=(-1.05, 0.11, 0.05),
        ),
        _decor_cube_cfg(
            "bed_headboard",
            size=(1.14, 0.06, 0.46),
            color=[0.28, 0.16, 0.09],
            position=(-1.05, 0.155, 0.25),
        ),
        _decor_cube_cfg(
            "bed_mattress",
            size=(0.98, 0.46, 0.08),
            color=[0.88, 0.79, 0.66],
            position=(-1.05, 0.08, 0.14),
        ),
        _decor_cube_cfg(
            "bed_duvet",
            size=(0.88, 0.34, 0.045),
            color=[0.62, 0.28, 0.17],
            position=(-1.05, 0.005, 0.205),
        ),
        _decor_cube_cfg(
            "bed_pillow_left",
            size=(0.33, 0.10, 0.055),
            color=[0.96, 0.86, 0.72],
            position=(-1.25, 0.22, 0.22),
        ),
        _decor_cube_cfg(
            "bed_pillow_right",
            size=(0.33, 0.10, 0.055),
            color=[0.96, 0.86, 0.72],
            position=(-0.85, 0.22, 0.22),
        ),
        _decor_cube_cfg(
            "left_nightstand",
            size=(0.24, 0.20, 0.22),
            color=[0.22, 0.12, 0.07],
            position=(-1.48, 0.03, 0.11),
        ),
        _decor_cube_cfg(
            "right_nightstand",
            size=(0.24, 0.20, 0.22),
            color=[0.22, 0.12, 0.07],
            position=(-0.43, 0.03, 0.11),
        ),
        _decor_cube_cfg(
            "bedside_lamp_base",
            size=(0.07, 0.07, 0.13),
            color=[0.70, 0.46, 0.20],
            position=(-0.43, 0.02, 0.285),
        ),
        _decor_cube_cfg(
            "bedside_lamp_shade",
            size=(0.16, 0.16, 0.12),
            color=[0.96, 0.67, 0.34],
            position=(-0.43, 0.02, 0.41),
        ),
        _decor_cube_cfg(
            "wardrobe",
            size=(0.44, 0.16, 0.88),
            color=[0.30, 0.17, 0.09],
            position=(1.34, 0.06, 0.44),
        ),
        _decor_cube_cfg(
            "window_glow_panel",
            size=(0.58, 0.014, 0.36),
            color=[1.0, 0.62, 0.30],
            position=(0.72, 0.151, 0.76),
        ),
        _decor_cube_cfg(
            "window_frame_top",
            size=(0.68, 0.02, 0.04),
            color=[0.23, 0.13, 0.08],
            position=(0.72, 0.142, 0.96),
        ),
        _decor_cube_cfg(
            "window_frame_bottom",
            size=(0.68, 0.02, 0.04),
            color=[0.23, 0.13, 0.08],
            position=(0.72, 0.142, 0.56),
        ),
        _decor_cube_cfg(
            "window_frame_left",
            size=(0.04, 0.02, 0.42),
            color=[0.23, 0.13, 0.08],
            position=(0.36, 0.142, 0.76),
        ),
        _decor_cube_cfg(
            "window_frame_right",
            size=(0.04, 0.02, 0.42),
            color=[0.23, 0.13, 0.08],
            position=(1.08, 0.142, 0.76),
        ),
    ]


def _indoor_light_cfgs() -> list[BaseLightCfg]:
    return [
        DomeLightCfg(
            name="indoor_warm_ambient",
            intensity=28.0,
            color=(0.95, 0.58, 0.26),
        ),
        DiskLightCfg(
            name="indoor_ceiling_key",
            intensity=360.0,
            radius=0.85,
            pos=(0.15, -0.55, 1.34),
            color=(1.0, 0.68, 0.32),
        ),
        SphereLightCfg(
            name="desk_lamp_warm_accent",
            intensity=48.0,
            radius=0.08,
            pos=(-0.43, -0.82, 0.52),
            color=(1.0, 0.48, 0.16),
        ),
    ]


def _extra_object_cfgs(task_spec: BenchmarkTaskSpec, simulator: str) -> list[object]:
    if _uses_indoor_scene(task_spec, simulator):
        return [
            *_indoor_scene_cfgs(),
            _elegant_glass_bottle_cfg(),
            _rigid_mjcf_prop_cfg("crystal_tumbler", CRYSTAL_TUMBLER_POS_M),
            _rigid_mjcf_prop_cfg("glass_hurricane_holder", GLASS_HURRICANE_POS_M),
            _articulated_mjcf_prop_cfg("articulated_clear_display_case", DISPLAY_CASE_POS_M),
            _articulated_mjcf_prop_cfg("articulated_desk_lamp", DESK_LAMP_POS_M),
        ]
    return []


def _camera_cfgs(task_spec: BenchmarkTaskSpec, camera_names: tuple[str, ...] | None) -> list[PinholeCameraCfg]:
    selected = task_spec.scene.camera_presets
    if camera_names is not None:
        requested = set(camera_names)
        selected = tuple(camera for camera in selected if camera.name in requested)
        missing = [name for name in camera_names if name not in {camera.name for camera in selected}]
        if missing:
            raise ValueError(f"unknown camera preset(s): {', '.join(missing)}")

    return [
        PinholeCameraCfg(
            name=camera.name,
            data_types=["rgb"],
            width=camera.width,
            height=camera.height,
            pos=camera.pos,
            look_at=camera.look_at,
            focal_length=24.0,
            horizontal_aperture=24.0,
        )
        for camera in selected
    ]


def build_benchmark_scenario(
    task_spec: BenchmarkTaskSpec,
    *,
    robot: str,
    simulator: str,
    headless: bool = True,
    camera_names: tuple[str, ...] | None = None,
) -> ScenarioCfg:
    """Build a native MetaSim scenario for a benchmark task and robot."""
    robot_cfg = get_robot(robot)
    scenario = ScenarioCfg(
        robots=[robot_cfg],
        objects=[
            _table_cfg(),
            *[_primitive_cube_cfg(obj) for obj in task_spec.scene.objects],
            *_extra_object_cfgs(task_spec, simulator),
        ],
        cameras=_camera_cfgs(task_spec, camera_names),
        simulator=simulator,
        headless=headless,
        env_spacing=3.0,
        decimation=17 if simulator == "mujoco" else 2,
    )
    if _uses_indoor_scene(task_spec, simulator):
        scenario.lights = _indoor_light_cfgs()
    return scenario


def _apply_dm_control_struct_compat_patch() -> None:
    from metasim.sim.mujoco.mujoco import apply_dm_control_struct_compat_patch

    apply_dm_control_struct_compat_patch()


def _get_handler_with_mujoco_compat(scenario: ScenarioCfg):
    if scenario.simulator == "mujoco":
        if getattr(scenario, "headless", False):
            os.environ.setdefault("MUJOCO_GL", "egl")
        _apply_dm_control_struct_compat_patch()
    return get_handler(scenario)


def _has_indoor_scene(scenario: ScenarioCfg) -> bool:
    return any(getattr(obj, "name", None) == "indoor_floor" for obj in scenario.objects)


def _add_blender_indoor_lights_if_supported(scenario: ScenarioCfg) -> None:
    if scenario.simulator != "blender" or not _has_indoor_scene(scenario):
        return

    import bpy

    scene = bpy.context.scene
    scene.view_settings.exposure = -1.25
    if scene.world is not None:
        scene.world.color = (0.008, 0.007, 0.006)

    for name, energy in (
        ("metasim_key_area_light", 45.0),
        ("metasim_fill_area_light", 10.0),
        ("metasim_rim_area_light", 14.0),
    ):
        obj = bpy.data.objects.get(name)
        if obj is not None and hasattr(obj, "data"):
            obj.data.energy = energy

    def add_area(name: str, location: tuple[float, float, float], energy: float, size: float) -> None:
        obj = bpy.data.objects.get(name)
        if obj is None:
            bpy.ops.object.light_add(type="AREA", location=location)
            obj = bpy.context.object
            obj.name = name
        obj.data.energy = energy
        obj.data.size = size
        obj.data.color = (1.0, 0.68, 0.32)
        if hasattr(obj, "visible_camera"):
            obj.visible_camera = False

    def add_point(name: str, location: tuple[float, float, float], energy: float, radius: float) -> None:
        obj = bpy.data.objects.get(name)
        if obj is None:
            bpy.ops.object.light_add(type="POINT", location=location)
            obj = bpy.context.object
            obj.name = name
        obj.data.energy = energy
        obj.data.shadow_soft_size = radius
        obj.data.color = (1.0, 0.48, 0.16)
        if hasattr(obj, "visible_camera"):
            obj.visible_camera = False

    add_area("indoor_ceiling_key", (0.15, -0.55, 1.34), 34.0, 0.85)
    add_point("desk_lamp_warm_accent", DESK_LAMP_POS_M[:2] + (0.52,), 8.0, 0.20)


def _maybe_to_device(value, device):
    if value is None or not hasattr(value, "to"):
        return value
    return value.to(device=device)


def _prepare_tensor_state_for_render_handler(physics_state, render_handler):
    render_device = getattr(render_handler, "device", None)
    if render_device is None:
        return physics_state

    objects = {
        name: replace(
            obj_state,
            root_state=_maybe_to_device(obj_state.root_state, render_device),
            body_state=_maybe_to_device(obj_state.body_state, render_device),
            joint_pos=_maybe_to_device(obj_state.joint_pos, render_device),
            joint_vel=_maybe_to_device(obj_state.joint_vel, render_device),
        )
        for name, obj_state in physics_state.objects.items()
    }
    robots = {
        name: replace(
            robot_state,
            root_state=_maybe_to_device(robot_state.root_state, render_device),
            body_state=_maybe_to_device(robot_state.body_state, render_device),
            joint_pos=_maybe_to_device(robot_state.joint_pos, render_device),
            joint_vel=_maybe_to_device(robot_state.joint_vel, render_device),
            joint_pos_target=_maybe_to_device(robot_state.joint_pos_target, render_device),
            joint_vel_target=_maybe_to_device(robot_state.joint_vel_target, render_device),
            joint_effort_target=_maybe_to_device(robot_state.joint_effort_target, render_device),
        )
        for name, robot_state in physics_state.robots.items()
    }
    extras = {
        key: _maybe_to_device(value, render_device) for key, value in getattr(physics_state, "extras", {}).items()
    }
    return replace(physics_state, objects=objects, robots=robots, cameras={}, extras=extras)


def _sync_render_from_physics(physics_handler, render_handler):
    physics_state = physics_handler.get_states(mode="tensor")
    try:
        render_handler.set_states(_prepare_tensor_state_for_render_handler(physics_state, render_handler))
    except TypeError:
        from metasim.utils.state import state_tensor_to_nested

        render_handler.set_states(state_tensor_to_nested(physics_handler, physics_state))

    if getattr(render_handler, "set_states_refreshes", False):
        return physics_state

    for hook_name in ("refresh_render", "render", "simulate", "update"):
        hook = getattr(render_handler, hook_name, None)
        if callable(hook):
            hook()
            break
    return physics_state


def _primary_robot_cfg(handler):
    if len(handler.robots) != 1:
        raise ValueError(f"expected a single robot configuration, got {len(handler.robots)}")
    return handler.robots[0]


def _canonical_control_joint_names(handler, robot_cfg) -> list[str]:
    available_joint_names = set(handler.get_joint_names(robot_cfg.name, sort=False))
    if getattr(robot_cfg, "default_joint_positions", None):
        ordered = [
            joint_name for joint_name in robot_cfg.default_joint_positions if joint_name in available_joint_names
        ]
        if ordered:
            return ordered
    if getattr(robot_cfg, "joint_limits", None):
        ordered = [joint_name for joint_name in robot_cfg.joint_limits if joint_name in available_joint_names]
        if ordered:
            return ordered
    return handler.get_joint_names(robot_cfg.name, sort=True)


def _expect_length(values: list[float], expected: int, *, label: str, robot_name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{label} length mismatch for {robot_name}: expected {expected}, got {len(values)}")


def _default_joint_targets(robot_cfg, joint_names: list[str]) -> list[float]:
    defaults = getattr(robot_cfg, "default_joint_positions", None)
    if defaults is None:
        raise ValueError(f"robot {robot_cfg.name} is missing default_joint_positions for hold")
    return [float(defaults[joint_name]) for joint_name in joint_names]


def _first_env_joint_values(joint_pos) -> list[float]:
    try:
        first_env = joint_pos[0]
    except Exception:
        first_env = joint_pos
    if hasattr(first_env, "detach"):
        first_env = first_env.detach().cpu().tolist()
    elif hasattr(first_env, "tolist"):
        first_env = first_env.tolist()
    return [float(value) for value in first_env]


def _current_joint_targets(handler, robot_cfg, joint_names: list[str]) -> tuple[float, ...]:
    states = handler.get_states(mode="tensor")
    robot_state = states.robots[robot_cfg.name]
    sorted_joint_names = handler.get_joint_names(robot_cfg.name, sort=True)
    joint_values = _first_env_joint_values(robot_state.joint_pos)
    _expect_length(joint_values, len(sorted_joint_names), label="current joint state", robot_name=robot_cfg.name)
    joint_value_by_name = {joint_name: joint_values[index] for index, joint_name in enumerate(sorted_joint_names)}
    return tuple(joint_value_by_name[joint_name] for joint_name in joint_names)


def _normalized_to_joint_target(normalized_value: float, default_value: float, lower: float, upper: float) -> float:
    clipped = max(-1.0, min(1.0, float(normalized_value)))
    action_scale = max(default_value - lower, upper - default_value, 1.0e-3)
    target = default_value + clipped * action_scale
    return max(lower, min(upper, target))


def control_command_to_metasim_actions(handler, command: ControlCommand):
    """Convert a teleop control command into MetaSim handler actions."""
    robot_cfg = _primary_robot_cfg(handler)
    joint_names = _canonical_control_joint_names(handler, robot_cfg)
    expected = len(joint_names)

    if command.mode == "hold":
        target_values = _default_joint_targets(robot_cfg, joint_names)
    elif command.mode == "joint_target_q":
        target_values = list(command.joint_target_q_rad or ())
        _expect_length(target_values, expected, label="joint target", robot_name=robot_cfg.name)
    elif command.mode == "normalized_action":
        normalized_values = list(command.normalized_action or ())
        _expect_length(normalized_values, expected, label="normalized action", robot_name=robot_cfg.name)
        joint_limits = getattr(robot_cfg, "joint_limits", None)
        default_joint_positions = getattr(robot_cfg, "default_joint_positions", None)
        if joint_limits is None or default_joint_positions is None:
            raise ValueError(f"robot {robot_cfg.name} is missing joint limits/defaults for normalized_action")
        target_values = [
            _normalized_to_joint_target(
                normalized_values[index],
                float(default_joint_positions[joint_name]),
                *joint_limits[joint_name],
            )
            for index, joint_name in enumerate(joint_names)
        ]
    else:
        raise ValueError(f"unsupported control mode: {command.mode}")

    return [
        {
            robot_cfg.name: {
                "dof_pos_target": {
                    joint_name: float(target_values[index]) for index, joint_name in enumerate(joint_names)
                },
                "dof_effort_target": None,
            }
        }
    ]


@dataclass
class NativeBenchmarkSession:
    """Native-only benchmark session wrapping one MetaSim handler."""

    scenario: ScenarioCfg

    def __post_init__(self) -> None:
        self.handler = _get_handler_with_mujoco_compat(self.scenario)
        _add_blender_indoor_lights_if_supported(self.scenario)
        self.backend_label = f"{self.scenario.simulator}+native"

    def reset(self):
        """Return the current handler state as the reset observation."""
        states = self.handler.get_states(mode="tensor")
        return states, {"backend": self.backend_label}

    def step(self, control: ControlCommand) -> StepResult:
        """Apply one teleop command and step native simulation."""
        self.handler.set_dof_targets(control_command_to_metasim_actions(self.handler, control))
        self.handler.simulate()
        refresh_render = getattr(self.handler, "refresh_render", None)
        if callable(refresh_render):
            refresh_render()
        return StepResult(
            observation=self.handler.get_states(mode="tensor"),
            info={"backend": self.backend_label, "controller": control.source},
        )

    def restore_state(self, state) -> StepResult:
        """Restore a previously captured state without advancing physics."""
        self.handler.set_states(state)
        refresh_render = getattr(self.handler, "refresh_render", None)
        if callable(refresh_render):
            refresh_render()
        return StepResult(
            observation=self.handler.get_states(mode="tensor"),
            info={"backend": self.backend_label, "controller": "script:initial_pose"},
        )

    def current_joint_targets(self) -> tuple[float, ...]:
        """Return current joints in the same order expected by joint-target commands."""
        robot_cfg = _primary_robot_cfg(self.handler)
        joint_names = _canonical_control_joint_names(self.handler, robot_cfg)
        return _current_joint_targets(self.handler, robot_cfg, joint_names)

    def render_state(self):
        """Return the current handler state, including camera data when available."""
        return self.handler.get_states(mode="tensor")

    def close(self) -> None:
        """Close the underlying MetaSim handler."""
        self.handler.close()


@dataclass
class HybridBenchmarkSession:
    """Benchmark session using separate MetaSim physics and render handlers."""

    physics_scenario: ScenarioCfg
    render_scenario: ScenarioCfg

    def __post_init__(self) -> None:
        self.physics_handler = _get_handler_with_mujoco_compat(self.physics_scenario)
        self.render_handler = _get_handler_with_mujoco_compat(self.render_scenario)
        _add_blender_indoor_lights_if_supported(self.render_scenario)
        self.backend_label = f"{self.physics_scenario.simulator}+{self.render_scenario.simulator}"

    def reset(self):
        """Return the current hybrid handler state as the reset observation."""
        states = _sync_render_from_physics(self.physics_handler, self.render_handler)
        return states, {"backend": self.backend_label}

    def step(self, control: ControlCommand) -> StepResult:
        """Apply one teleop command to physics and sync the render handler."""
        self.physics_handler.set_dof_targets(control_command_to_metasim_actions(self.physics_handler, control))
        self.physics_handler.simulate()
        states = _sync_render_from_physics(self.physics_handler, self.render_handler)
        return StepResult(
            observation=states,
            info={"backend": self.backend_label, "controller": control.source},
        )

    def restore_state(self, state) -> StepResult:
        """Restore physics state and sync the render handler without advancing physics."""
        self.physics_handler.set_states(state)
        states = _sync_render_from_physics(self.physics_handler, self.render_handler)
        return StepResult(
            observation=states,
            info={"backend": self.backend_label, "controller": "script:initial_pose"},
        )

    def current_joint_targets(self) -> tuple[float, ...]:
        """Return current physics joints in the same order expected by joint-target commands."""
        robot_cfg = _primary_robot_cfg(self.physics_handler)
        joint_names = _canonical_control_joint_names(self.physics_handler, robot_cfg)
        return _current_joint_targets(self.physics_handler, robot_cfg, joint_names)

    def render_state(self):
        """Return the current render handler state, including rendered camera data."""
        return self.render_handler.get_states(mode="tensor")

    def close(self) -> None:
        """Close both hybrid handlers."""
        try:
            self.physics_handler.close()
        finally:
            self.render_handler.close()


__all__ = [
    "HybridBenchmarkSession",
    "NativeBenchmarkSession",
    "build_benchmark_scenario",
    "control_command_to_metasim_actions",
]
