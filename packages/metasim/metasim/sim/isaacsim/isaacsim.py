# This naively suites for isaaclab 2.2.0 and isaacsim 5.0.0

from __future__ import annotations

import argparse
import gc
import math
import os
import threading
from copy import deepcopy
from importlib import resources

import numpy as np
import torch
from loguru import logger as log
from scipy.interpolate import RegularGridInterpolator

from metasim.queries.base import BaseQueryType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    BaseArticulationObjCfg,
    BaseObjCfg,
    BaseRigidObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveFrameCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import ActionBatch, CompatActionInput, DictEnvState
from metasim.utils.dict import deep_get
from metasim.utils.gs_util import alpha_blend_rgba_torch
from metasim.utils.isaacsim_asset_util import resolve_isaacsim_file_path
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, action_input_to_tensor
from metasim.utils.terrain_utils import TerrainGenerator

try:
    from robo_splatter.models.camera import Camera as SplatCamera
    from robo_splatter.render.scenes import SceneRenderType

    ROBO_SPLATTER_AVAILABLE = True
except ImportError:
    ROBO_SPLATTER_AVAILABLE = False
    log.warning("RoboSplatter not available. GS background rendering will be disabled.")


def _world_state_to_env_local(state_w: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    """Return an env-local copy of an IsaacLab world-frame state tensor."""
    local_state = state_w.clone()
    env_origins = env_origins.to(device=local_state.device, dtype=local_state.dtype)
    if local_state.ndim == 2:
        local_state[:, 0:3] -= env_origins
    elif local_state.ndim == 3:
        local_state[:, :, 0:3] -= env_origins[:, None, :]
    else:
        raise ValueError(f"Expected state tensor rank 2 or 3, got shape {tuple(local_state.shape)}")
    return local_state


def _primitive_frame_usd_path() -> str:
    """Return the packaged USD used for PrimitiveFrameCfg."""
    return str(resources.files("metasim").joinpath("data/quick_start/assets/COMMON/frame/usd/frame.usd"))


# RTX Real-Time 2.0 must be REGISTERED when Kit boots: the renderer only joins
# the hydra render-mode list when ``/rtx-transient/rt2Enabled`` is on, and Kit
# derives that at startup from the persistent preference
# ``/persistent/rtx/modes/rt2/enabled`` (a transient-only CLI flag is clobbered
# by that copy). Once registered, ``/rtx/rendermode`` switches in and out of
# RealTimePathTracing at RUNTIME just fine (probed on Isaac Sim 5.0) — without
# registration every write is silently refused, which is why a bare rendermode
# set (runtime, Kit boot argv, or the SimulationApp ``renderer`` config) never
# engages it.
_BOOT_KIT_ARGS_BY_RENDER_MODE = {
    "realtime_pathtracing": (
        "--/persistent/rtx/modes/rt2/enabled=true",
        "--/rtx-transient/rt2Enabled=true",
    ),
}


def _kit_args_with_boot_render_mode(render_mode: str | None, existing_kit_args: str | None) -> str | None:
    """AppLauncher ``kit_args`` engaging render modes that need boot-time registration.

    Returns ``existing_kit_args`` unchanged when ``render_mode`` needs nothing at
    boot; otherwise appends the missing registration flags (idempotent — flags
    already present are not duplicated). AppLauncher space-splits the string into
    ``sys.argv`` for Kit (``_resolve_kit_args``).
    """
    flags = _BOOT_KIT_ARGS_BY_RENDER_MODE.get(render_mode or "")
    if not flags:
        return existing_kit_args
    present = set(existing_kit_args.split()) if existing_kit_args else set()
    missing = [f for f in flags if f not in present]
    if not missing:
        return existing_kit_args
    return " ".join(([existing_kit_args] if existing_kit_args else []) + missing)


class IsaacsimHandler(BaseSimHandler):
    """
    Handler for Isaac Lab simulation environment.
    This class extends BaseSimHandler to provide specific functionality for Isaac Lab.
    """

    @property
    def set_states_refreshes(self) -> bool:  # type: ignore[override]
        """``_set_states`` ends with ``refresh_render()`` only when the scene has cameras."""
        return len(self.cameras) > 0

    def __init__(self, scenario_cfg: ScenarioCfg, optional_queries: list[BaseQueryType] | None = None):
        super().__init__(scenario_cfg, optional_queries)

        self._actions_cache: CompatActionInput = []
        self._robot_names = {robot.name for robot in self.robots}
        self._robot_init_pos = {robot.name: robot.default_position for robot in self.robots}
        self._robot_init_quat = {robot.name: robot.default_orientation for robot in self.robots}
        self._cameras = scenario_cfg.cameras

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_envs: int = scenario_cfg.num_envs
        self._episode_length_buf = [0 for _ in range(self.num_envs)]

        self.scenario_cfg = scenario_cfg
        # Calculate physics_dt to ensure dt * decimation = constant (0.015)
        if self.scenario.sim_params.dt is not None:
            self.physics_dt = self.scenario.sim_params.dt
        else:
            # Default: dt * decimation = 0.015
            self.physics_dt = 0.015 / self.scenario.decimation
        self._physics_step_counter = 0
        self._render_current = False  # set by refresh_render, cleared by _simulate
        self._is_closed = False
        self.render_interval = self.scenario.decimation  # TODO: fix hardcode
        self._manual_pd_on = []
        self._owns_simulation_app = False
        self.simulation_app = None
        self.scene = None
        self.sim = None
        self.contact_sensor = None

        if self.headless:
            self._render_viewport = False
        else:
            self._render_viewport = True

        self._sync_tensor_device_with_sim_device()

    def _resolve_sim_device(self) -> str:
        return os.environ.get("METASIM_EVAL_ISAACSIM_DEVICE", "cuda:0")

    def _sync_tensor_device_with_sim_device(self) -> None:
        self._device = torch.device(self._resolve_sim_device())

    def _init_scene(self, simulation_app=None, args=None) -> None:
        """
        Initializes the isaacsim simulation environment.
        """
        if simulation_app is None:
            from isaaclab.app import AppLauncher

            parser = argparse.ArgumentParser()
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.enable_cameras = bool(self.cameras)
            args.headless = self.headless
            args.device = self._resolve_sim_device()
            # Boot-time renderer REGISTRATION (RTX Real-Time 2.0): without the
            # rt2 enable flags in Kit's boot argv the renderer is never
            # registered and every /rtx/rendermode write is silently refused.
            # Once registered, _load_render_settings' runtime write engages it.
            kit_args = _kit_args_with_boot_render_mode(
                getattr(self.scenario.render, "mode", None), getattr(args, "kit_args", None)
            )
            if kit_args:
                args.kit_args = kit_args
            app_launcher = AppLauncher(args)
            self.simulation_app = app_launcher.app
            self._owns_simulation_app = True
        else:
            self.simulation_app = simulation_app
            self._owns_simulation_app = False

        # physics context
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

        sim_device = self._resolve_sim_device()
        self._sync_tensor_device_with_sim_device()
        sim_config: SimulationCfg = SimulationCfg(
            device=sim_device,
            render_interval=self.scenario.decimation,  # TODO divide into render interval and control decimation
            physx=PhysxCfg(
                bounce_threshold_velocity=self.scenario.sim_params.bounce_threshold_velocity,
                solver_type=self.scenario.sim_params.solver_type,
                max_position_iteration_count=self.scenario.sim_params.num_position_iterations,
                max_velocity_iteration_count=self.scenario.sim_params.num_velocity_iterations,
                friction_correlation_distance=self.scenario.sim_params.friction_correlation_distance,
                friction_offset_threshold=self.scenario.sim_params.friction_offset_threshold,
            ),
            dt=self.physics_dt,
        )

        self.sim: SimulationContext = SimulationContext(sim_config)
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self._num_envs, env_spacing=self.scenario.env_spacing
        )
        self.scene = InteractiveScene(scene_config)

        if self.sim.has_gui():
            self._init_keyboard()

    def _load_robots(self) -> None:
        for robot in self.robots:
            self._add_robot(robot)

    def _load_objects(self) -> None:
        for obj_cfg in self.objects:
            self._add_object(obj_cfg)

    def _load_cameras(self) -> None:
        for camera in self.cameras:
            if isinstance(camera, PinholeCameraCfg):
                self._add_pinhole_camera(camera)
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def _init_keyboard(self) -> None:
        import weakref

        import carb
        import omni

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        obj_proxy = weakref.proxy(self)
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args: obj_proxy._on_keyboard_event(event, *args),
        )

    def _update_camera_pose(self) -> None:
        env_origins = getattr(self.scene, "env_origins", None)
        if env_origins is None:
            env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            env_origins = env_origins.to(self.device)

        for camera in self.cameras:
            if isinstance(camera, PinholeCameraCfg):
                # set look at position using isaaclab's api
                if camera.mount_to is None:
                    camera_inst = self.scene.sensors[camera.name]
                    position_tensor = torch.as_tensor(camera.pos, device=self.device).expand(self.num_envs, -1)
                    camera_lookat_tensor = torch.as_tensor(camera.look_at, device=self.device).expand(self.num_envs, -1)
                    position_tensor = position_tensor + env_origins
                    camera_lookat_tensor = camera_lookat_tensor + env_origins
                    camera_inst.set_world_poses_from_view(position_tensor, camera_lookat_tensor)
                    # log.debug(f"Updated camera {camera.name} pose: pos={camera.pos}, look_at={camera.look_at}")
            else:
                raise ValueError(f"Unsupported camera type: {type(camera)}")

    def _apply_robot_default_joint_positions(self) -> None:
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        for robot in self.robots:
            robot_inst = self.scene.articulations[robot.name]
            joint_names = list(robot_inst.joint_names)
            joint_pos = robot_inst.data.default_joint_pos[env_ids].clone()
            for joint_index, joint_name in enumerate(joint_names):
                if joint_name in robot.default_joint_positions:
                    joint_pos[:, joint_index] = float(robot.default_joint_positions[joint_name])
            joint_vel = torch.zeros_like(joint_pos)
            robot_inst.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot_inst.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            robot_inst.write_data_to_sim()

    def _initialize_physics_before_reset(self) -> None:
        initialize_physics = getattr(self.sim, "initialize_physics", None)
        if callable(initialize_physics):
            initialize_physics()

    def launch(self, simulation_app=None, simulation_args=None) -> None:
        try:
            self._launch_impl(simulation_app, simulation_args)
        except Exception as err:
            # A failure anywhere during launch (asset load, physics reset, or
            # camera-sensor init) otherwise leaves the Omniverse Kit
            # SimulationApp alive with non-daemon background threads. The Python
            # main thread dies on the exception but the process never exits, so
            # it "hangs" while still holding the GPU. Tear the app down here so
            # the error surfaces and the GPU is freed, then re-raise.
            self._on_launch_failure(err)
            raise

    def _launch_impl(self, simulation_app=None, simulation_args=None) -> None:
        self._init_scene(simulation_app, simulation_args)
        self._load_robots()
        self._load_sensors()
        if self.scenario.scene is None:
            self._load_terrain()
        self._load_scene()
        self._load_objects()
        self._load_lights()
        self._load_render_settings()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self._load_cameras()
        self._initialize_physics_before_reset()
        self.sim.reset()
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.scene.reset(indices)

        self._apply_robot_default_joint_positions()

        # Update camera pose after scene reset to avoid being overridden
        self._update_camera_pose()

        # Force another simulation step and camera update to ensure proper initialization
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._update_camera_pose()

        # Force a render to update camera data after position is set
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)

        # Initialize GS background if enabled
        self._build_gs_background()
        super().launch()
        for sensor in self.scene.sensors.values():
            if hasattr(sensor, "_initialize_callback"):
                sensor._initialize_callback(None)

    def _on_launch_failure(self, err: Exception) -> None:
        """Free the GPU after a launch-time failure so the process cannot hang.

        Only tears down a SimulationApp this handler owns; a caller-provided
        shared app is left untouched. ``close()`` carries its own force-exit
        watchdog (``METASIM_FORCE_EXIT_ON_CLOSE``), so a wedged Kit shutdown
        still exits instead of hanging.
        """
        log.error(f"IsaacSim launch failed ({type(err).__name__}: {err}); closing SimulationApp to free the GPU.")
        self._maybe_hint_numpy_abi(err)
        if getattr(self, "_owns_simulation_app", False) and getattr(self, "simulation_app", None) is not None:
            try:
                self.close()
            except Exception as close_err:
                log.debug(f"close() during launch-failure teardown raised: {close_err}")

    @staticmethod
    def _maybe_hint_numpy_abi(err: Exception) -> None:
        """Surface the numpy<2 requirement when the failure looks like an ABI mismatch.

        IsaacSim 5.0 / Isaac Lab are built against numpy<2. With numpy>=2 the
        ``omni.syntheticdata`` camera-annotation graph fails to wire up with
        ``TypeError: Unable to write from unknown dtype`` (preceded by
        ``missing valid input renderVar LdrColorSD``), which manifests as a
        broken camera render rather than an obvious version error.
        """
        text = " ".join(str(part) for part in (err, getattr(err, "__cause__", None)) if part is not None)
        if "unknown dtype" not in text and "renderVar" not in text:
            return
        try:
            import numpy as _np

            if int(_np.__version__.split(".")[0]) < 2:
                return
            version = _np.__version__
        except Exception:
            version = ">=2"
        log.error(
            f"This looks like a numpy ABI mismatch: numpy {version} is installed but IsaacSim/Isaac Lab "
            "require numpy<2. Install a numpy<2 build (e.g. `pip install 'numpy<2'`) in this environment "
            "to restore IsaacSim camera rendering."
        )

    def close(self) -> None:
        log.info("close Isaacsim Handler")
        if not self._is_closed:
            # Default ON: IsaacSim's Kit-app shutdown (Replicator workers, hydra
            # delegates, GPU teardown) is known to hang upstream. Set
            # METASIM_FORCE_EXIT_ON_CLOSE=0 to disable, e.g. if you call close()
            # between episodes and need post-close code to run.
            force_exit_on_close_hang = os.getenv("METASIM_FORCE_EXIT_ON_CLOSE", "1") == "1"
            close_timeout_s = float(os.getenv("METASIM_CLOSE_TIMEOUT_SEC", "15"))
            simulation_app = getattr(self, "simulation_app", None)
            owns_simulation_app = getattr(self, "_owns_simulation_app", False)

            if self.sim is not None:
                try:
                    # Match IsaacLab's own test teardown pattern before
                    # clearing the singleton simulation context.
                    self.sim._disable_app_control_on_stop_handle = True
                except Exception:
                    pass

                stop_fn = getattr(self.sim, "stop", None)
                if callable(stop_fn):
                    try:
                        stop_fn()
                    except Exception as err:
                        log.debug(f"SimulationContext.stop() failed during close: {err}")
                else:
                    timeline = getattr(self.sim, "_timeline", None)
                    if timeline is not None:
                        try:
                            timeline.stop()
                        except Exception as err:
                            log.debug(f"SimulationContext timeline.stop() failed during close: {err}")

                clear_callbacks = getattr(self.sim, "clear_all_callbacks", None)
                if callable(clear_callbacks):
                    try:
                        clear_callbacks()
                    except Exception as err:
                        log.debug(f"SimulationContext.clear_all_callbacks() failed during close: {err}")

                clear_instance = getattr(self.sim, "clear_instance", None)
                if callable(clear_instance):
                    try:
                        clear_instance()
                    except Exception as err:
                        log.debug(f"SimulationContext.clear_instance() failed during close: {err}")

            if simulation_app is not None and force_exit_on_close_hang and owns_simulation_app:
                close_error: dict[str, Exception] = {}

                def _close_sim_app() -> None:
                    try:
                        # Isaac Sim may hang on shutdown waiting for Replicator workers.
                        simulation_app.close(wait_for_replicator=False)
                    except TypeError:
                        # Backward compatibility with versions where close() has no kwargs.
                        simulation_app.close()
                    except Exception as err:
                        close_error["error"] = err

                close_thread = threading.Thread(target=_close_sim_app, daemon=True)
                close_thread.start()
                close_thread.join(timeout=close_timeout_s)
                if close_thread.is_alive():
                    log.warning(
                        f"SimulationApp.close() exceeded {close_timeout_s:.1f}s. "
                        "Forcing process exit to avoid shutdown hang."
                    )
                    import sys

                    sys.stdout.flush()
                    sys.stderr.flush()
                    # The simulation itself finished; only Kit's shutdown hung (it does on
                    # every Isaac Sim 5.0 run here). Exit 0 so scripts and test matrices
                    # report the run's real outcome; METASIM_ISAACSIM_HANG_EXIT_CODE
                    # restores a distinguishable status for pipelines that want one. A
                    # close() that runs while an exception is propagating (a failed
                    # launch, a context manager unwinding an error) must still fail.
                    raw = os.environ.get("METASIM_ISAACSIM_HANG_EXIT_CODE", "0")
                    try:
                        exit_code = int(raw)
                    except ValueError:
                        log.error(f"METASIM_ISAACSIM_HANG_EXIT_CODE={raw!r} is not an integer; exiting 1")
                        exit_code = 1
                    if sys.exc_info()[0] is not None:
                        log.error(f"exiting 1: {sys.exc_info()[0].__name__} was propagating when close() hung")
                        exit_code = 1
                    os._exit(exit_code)
                if "error" in close_error:
                    raise close_error["error"]
            elif simulation_app is not None and owns_simulation_app:
                try:
                    simulation_app.close(wait_for_replicator=False)
                except TypeError:
                    simulation_app.close()
            if self.contact_sensor is not None:
                try:
                    clear_callbacks = getattr(self.contact_sensor, "_clear_callbacks", None)
                    if callable(clear_callbacks):
                        clear_callbacks()
                    invalidate = getattr(self.contact_sensor, "_invalidate_initialize_callback", None)
                    if callable(invalidate):
                        invalidate(None)
                    for attr_name in (
                        "_body_physx_view",
                        "_contact_physx_view",
                        "_physics_sim_view",
                        "_data",
                        "_is_outdated",
                        "_timestamp",
                        "_timestamp_last_update",
                        "stage",
                    ):
                        if hasattr(self.contact_sensor, attr_name):
                            setattr(self.contact_sensor, attr_name, None)
                except Exception as err:
                    log.debug(f"Contact sensor cleanup failed during close: {err}")
                if self.scene is not None:
                    try:
                        self.scene.sensors.pop("contact_sensor", None)
                    except Exception as err:
                        log.debug(f"Failed to detach contact sensor from scene: {err}")
            if self.scene is not None:
                del self.scene
            if self.sim is not None:
                del self.sim
            if self.simulation_app is not None:
                del self.simulation_app

            # Drop handler-owned references that can otherwise keep IsaacSim
            # sensor / stage resources alive across suite-local handler restarts.
            if hasattr(self, "contact_sensor"):
                self.contact_sensor = None
            if hasattr(self, "terrain"):
                self.terrain = None
            if hasattr(self, "_ground_mesh_vertices"):
                self._ground_mesh_vertices = None
            if hasattr(self, "_ground_mesh_triangles"):
                self._ground_mesh_triangles = None
            if hasattr(self, "_height_mat"):
                self._height_mat = None
            if hasattr(self, "_appwindow"):
                self._appwindow = None
            if hasattr(self, "_input"):
                self._input = None
            if hasattr(self, "_keyboard"):
                self._keyboard = None
            if hasattr(self, "_keyboard_sub"):
                self._keyboard_sub = None

            if simulation_app is not None and not owns_simulation_app:
                try:
                    import isaacsim.core.utils.stage as stage_utils

                    stage_utils.close_stage()
                except Exception as err:
                    log.debug(f"Stage close failed during shared-app cleanup: {err}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_closed = True

    def _set_states(self, states: list[DictEnvState] | TensorState, env_ids: list[int] | None = None) -> None:
        # if states is list[DictEnvState], iterate over it and set state
        if isinstance(states, list):
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            # Handle different state list lengths:
            # 1. Single state -> replicate across all envs (most common for initial setup)
            # 2. States matching num_envs -> use corresponding state per env
            if len(states) == 1:
                # Replicate single state across all environments
                states_flat = [states[0]["objects"] | states[0]["robots"] for _ in range(self.num_envs)]
            elif len(states) == self.num_envs:
                # Use provided states for each environment
                states_flat = [states[i]["objects"] | states[i]["robots"] for i in range(self.num_envs)]
            else:
                raise ValueError(
                    f"States list length ({len(states)}) must be either 1 (replicate to all envs) "
                    f"or match num_envs ({self.num_envs}). Got {len(states)} states."
                )
            for obj in self.objects + self.robots:
                if obj.name not in states_flat[0]:
                    log.warning(f"Missing {obj.name} in states, setting its velocity to zero")
                    pos, rot = self._get_pose(obj.name, env_ids=env_ids)
                    self._set_object_pose(obj, pos, rot, env_ids=env_ids)
                    continue

                if (
                    states_flat[0][obj.name].get("pos", None) is None
                    or states_flat[0][obj.name].get("rot", None) is None
                ):
                    log.warning(f"No pose found for {obj.name}, setting its velocity to zero")
                    pos, rot = self._get_pose(obj.name, env_ids=env_ids)
                    self._set_object_pose(obj, pos, rot, env_ids=env_ids)
                else:
                    # Coerce list / tuple / numpy / torch inputs into a stacked
                    # tensor on self.device. ``torch.stack`` itself requires
                    # tensor elements — without ``torch.as_tensor`` here, callers
                    # that pass plain Python lists (which mujoco / sapien3 /
                    # newton accept fine) hit ``TypeError: expected Tensor as
                    # element 0`` and the cross-backend round-trip contract is
                    # silently broken on isaacsim only.
                    pos = torch.stack([
                        torch.as_tensor(states_flat[env_id][obj.name]["pos"], device=self.device) for env_id in env_ids
                    ])
                    rot = torch.stack([
                        torch.as_tensor(states_flat[env_id][obj.name]["rot"], device=self.device) for env_id in env_ids
                    ])
                    self._set_object_pose(obj, pos, rot, env_ids=env_ids)

                if isinstance(obj, ArticulationObjCfg):
                    if states_flat[0][obj.name].get("dof_pos", None) is None:
                        log.warning(f"No dof_pos found for {obj.name}")
                    else:
                        dof_dict = [states_flat[env_id][obj.name]["dof_pos"] for env_id in env_ids]
                        joint_names = self._get_joint_names(obj.name, sort=False)
                        joint_pos = torch.zeros((len(env_ids), len(joint_names)), device=self.device)
                        for i, joint_name in enumerate(joint_names):
                            if joint_name in dof_dict[0]:
                                joint_pos[:, i] = torch.tensor([x[joint_name] for x in dof_dict], device=self.device)
                            else:
                                log.warning(f"Missing {joint_name} in {obj.name}, setting its position to zero")

                        self._set_object_joint_pos(obj, joint_pos, env_ids=env_ids)
                        if obj in self.robots:
                            robot_inst = self.scene.articulations[obj.name]
                            robot_inst.set_joint_position_target(
                                joint_pos, env_ids=torch.tensor(env_ids, device=self.device)
                            )
                            robot_inst.write_data_to_sim()

            if len(self.cameras) > 0:
                self.refresh_render()

        # if states is TensorState, reindex the tensors and set state
        elif isinstance(states, TensorState):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            elif isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device)

            for _, obj in enumerate(self.objects):
                if isinstance(obj, ArticulationObjCfg):
                    obj_inst = self.scene.articulations[obj.name]
                else:
                    obj_inst = self.scene.rigid_objects[obj.name]

                # Set root state (fix_base_link only affects physics, not manual state setting)
                root_state = states.objects[obj.name].root_state.clone()
                root_state[:, :3] += self.scene.env_origins
                obj_inst.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
                obj_inst.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
                # Set joint state for articulated objects
                if isinstance(obj, ArticulationObjCfg):
                    joint_ids_reindex = self.get_joint_reindex(obj.name, inverse=True)
                    obj_inst.write_joint_position_to_sim(
                        states.objects[obj.name].joint_pos[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                    )
                    obj_inst.write_joint_velocity_to_sim(
                        states.objects[obj.name].joint_vel[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                    )
                write_data = getattr(obj_inst, "write_data_to_sim", None)
                if callable(write_data):
                    write_data()

                # For kinematic objects (fix_base_link=True), force update to sync visual mesh
                if obj.fix_base_link:
                    obj_inst.update(dt=0.0)

            for _, robot in enumerate[RobotCfg](self.robots):
                robot_inst = self.scene.articulations[robot.name]
                root_state = states.robots[robot.name].root_state.clone()
                root_state[:, :3] += self.scene.env_origins
                robot_inst.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
                robot_inst.write_root_velocity_to_sim(
                    states.robots[robot.name].root_state[env_ids, 7:], env_ids=env_ids
                )
                joint_ids_reindex = self.get_joint_reindex(robot.name, inverse=True)
                robot_inst.write_joint_position_to_sim(
                    states.robots[robot.name].joint_pos[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                )
                robot_inst.write_joint_velocity_to_sim(
                    states.robots[robot.name].joint_vel[env_ids, :][:, joint_ids_reindex], env_ids=env_ids
                )
                joint_pos = states.robots[robot.name].joint_pos[env_ids, :][:, joint_ids_reindex]
                robot_inst.set_joint_position_target(joint_pos, env_ids=env_ids)
                robot_inst.write_data_to_sim()

            if len(self.cameras) > 0:
                self._update_camera_pose()
                self.refresh_render()
        else:
            raise Exception("Unsupported state type, must be DictEnvState or TensorState")

    def _get_foreground_mask(
        self,
        instance_seg_data: torch.Tensor | None,
        instance_seg_id2label: dict[int, str] | None,
        instance_id_seg_data: torch.Tensor | None,
        instance_id_seg_id2label: dict[int, str] | None,
    ) -> torch.Tensor | None:
        """
        Create foreground mask by excluding terrain/ground from instance segmentation data.

        Args:
            instance_seg_data: Instance segmentation data (semantic level).
            instance_seg_id2label: Mapping from instance IDs to labels for instance_seg_data.
            instance_id_seg_data: Instance ID segmentation data (instance level, more precise).
            instance_id_seg_id2label: Mapping from instance IDs to labels for instance_id_seg_data.

        Returns:
            Foreground mask tensor: 1 for objects (not terrain), 0 for terrain/background.
            Returns None if no instance segmentation data is available.
        """
        foreground_mask = None

        # Use instance_id_seg_data if available (more precise), otherwise use instance_seg_data
        if instance_id_seg_data is not None and instance_id_seg_id2label is not None:
            # Find terrain IDs from labels
            terrain_ids = {
                id
                for id, label in instance_id_seg_id2label.items()
                if any(kw in label.lower() for kw in ["ground", "terrain", "floor", "world/ground"])
            }
            unique_ids = torch.unique(instance_id_seg_data)

            if terrain_ids:
                # Object mask: 1 for objects (not terrain), 0 for terrain/background
                foreground_mask = torch.ones_like(instance_id_seg_data, dtype=torch.float32)
                for terrain_id in terrain_ids:
                    if terrain_id in unique_ids:
                        foreground_mask[instance_id_seg_data == terrain_id] = 0.0
                # Exclude background (id == 0) if it exists
                if 0 in unique_ids:
                    foreground_mask[instance_id_seg_data == 0] = 0.0
        elif instance_seg_data is not None and instance_seg_id2label is not None:
            # Fallback to instance_seg_data
            terrain_ids = {
                id
                for id, label in instance_seg_id2label.items()
                if any(kw in label.lower() for kw in ["ground", "terrain", "floor", "world/ground"])
            }
            unique_ids = torch.unique(instance_seg_data)

            if terrain_ids:
                foreground_mask = torch.ones_like(instance_seg_data, dtype=torch.float32)
                for terrain_id in terrain_ids:
                    if terrain_id in unique_ids:
                        foreground_mask[instance_seg_data == terrain_id] = 0.0
                # Exclude background (id == 0) if it exists
                if 0 in unique_ids:
                    foreground_mask[instance_seg_data == 0] = 0.0

        # Fallback: if no terrain IDs found or mask is all zeros, use simple foreground mask
        if foreground_mask is None or (foreground_mask is not None and foreground_mask.sum() == 0):
            if instance_id_seg_data is not None:
                foreground_mask = (instance_id_seg_data > 0).float()
            elif instance_seg_data is not None:
                foreground_mask = (instance_seg_data > 0).float()
            else:
                log.warning("No instance segmentation data available for foreground mask")

        return foreground_mask

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Special handling for the first frame to ensure camera is properly positioned; skipped when
        # refresh_render already rendered this state (the hybrid handler renders every synced step)
        if self._physics_step_counter == 0 and not self._render_current:
            self._update_camera_pose()
            # Force render and sensor update for first frame
            if self.sim.has_gui() or self.sim.has_rtx_sensors():
                self.sim.render()
            for sensor in self.scene.sensors.values():
                sensor.update(dt=0)

        object_states = {}
        for obj in self.objects:
            if isinstance(obj, ArticulationObjCfg):
                obj_inst = self.scene.articulations[obj.name]
                joint_reindex = self.get_joint_reindex(obj.name)
                body_reindex = self.get_body_reindex(obj.name)
                root_state = _world_state_to_env_local(obj_inst.data.root_state_w, self.scene.env_origins)
                body_state = _world_state_to_env_local(
                    obj_inst.data.body_state_w[:, body_reindex], self.scene.env_origins
                )
                state = ObjectState(
                    root_state=root_state,
                    body_names=self._get_body_names(obj.name),
                    body_state=body_state,
                    joint_pos=obj_inst.data.joint_pos[:, joint_reindex].clone(),
                    joint_vel=obj_inst.data.joint_vel[:, joint_reindex].clone(),
                )
            else:
                obj_inst = self.scene.rigid_objects[obj.name]
                root_state = _world_state_to_env_local(obj_inst.data.root_state_w, self.scene.env_origins)
                state = ObjectState(
                    root_state=root_state,
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in self.robots:
            obj_inst = self.scene.articulations[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
            body_reindex = self.get_body_reindex(obj.name)
            root_state = _world_state_to_env_local(obj_inst.data.root_state_w, self.scene.env_origins)
            body_state = _world_state_to_env_local(obj_inst.data.body_state_w[:, body_reindex], self.scene.env_origins)
            state = RobotState(
                root_state=root_state,
                body_names=self._get_body_names(obj.name),
                body_state=body_state,
                joint_pos=obj_inst.data.joint_pos[:, joint_reindex].clone(),
                joint_vel=obj_inst.data.joint_vel[:, joint_reindex].clone(),
                joint_pos_target=obj_inst.data.joint_pos_target[:, joint_reindex].clone(),
                joint_vel_target=obj_inst.data.joint_vel_target[:, joint_reindex].clone(),
                joint_effort_target=obj_inst.data.joint_effort_target[:, joint_reindex].clone(),
            )
            robot_states[obj.name] = state

        camera_states = {}
        # Force camera sensor update to ensure correct position data
        for sensor in self.scene.sensors.values():
            sensor.update(dt=0)

        for camera in self.cameras:
            camera_inst = self.scene.sensors[camera.name]
            rgb_data = camera_inst.data.output.get("rgb", None)
            depth_data = camera_inst.data.output.get("depth", None)
            camera_info = self._camera_info_mapping(camera_inst)
            instance_seg_data = deep_get(camera_inst.data.output, "instance_segmentation_fast")
            instance_seg_id2label = deep_get(camera_info, "instance_segmentation_fast", "idToLabels")
            instance_id_seg_data = deep_get(camera_inst.data.output, "instance_id_segmentation_fast")
            instance_id_seg_id2label = deep_get(camera_info, "instance_id_segmentation_fast", "idToLabels")
            if instance_seg_data is not None:
                instance_seg_data = instance_seg_data.squeeze(-1)
            if instance_id_seg_data is not None:
                instance_id_seg_data = instance_id_seg_data.squeeze(-1)
            # Depth comes in as ``(envs, H, W, 1)``; ``CameraState.depth`` is
            # documented as rank 3 (``(envs, H, W)``). Match the existing
            # instance-seg squeeze convention so ``CameraState`` validation
            # doesn't reject the tensor. The GS-blending branch below
            # rebuilds depth_data with its own ``unsqueeze(-1)`` so we
            # don't have to track shapes through that path either.
            if depth_data is not None and depth_data.ndim == 4 and depth_data.shape[-1] == 1:
                depth_data = depth_data.squeeze(-1)

            # GS background blending
            if (
                self.scenario.gs_scene is not None
                and self.scenario.gs_scene.with_gs_background
                and rgb_data is not None
            ):
                assert ROBO_SPLATTER_AVAILABLE, (
                    "RoboSplatter is not available. GS background rendering will be disabled."
                )

                foreground_mask = self._get_foreground_mask(
                    instance_seg_data, instance_seg_id2label, instance_id_seg_data, instance_id_seg_id2label
                )
                assert foreground_mask is not None, "Foreground mask is None"

                # Get camera parameters (already as torch tensors on device)
                Ks_t, c2w_t = self._get_camera_params(camera, camera_inst)

                # Create GS camera and render
                gs_cam = SplatCamera.init_from_pose_tensor(
                    c2w=c2w_t,
                    Ks=Ks_t,
                    image_height=int(camera.height),
                    image_width=int(camera.width),
                    device=self.device,
                )

                gs_result = self.gs_background.render(gs_cam, render_type=SceneRenderType.FOREGROUND)

                # Get RGB Blending with GS background
                sim_rgb = rgb_data.float() / 255.0  # Normalize to [0, 1], Shape: (envs, H, W, 3)
                gs_rgb = gs_result.rgb  # Shape: (envs, H, W, 3), BGR order

                if isinstance(gs_rgb, np.ndarray):
                    gs_rgb = torch.from_numpy(gs_rgb)
                gs_rgb = gs_rgb.to(self.device)
                blended_rgb = alpha_blend_rgba_torch(sim_rgb, gs_rgb, foreground_mask)
                rgb_data = (blended_rgb * 255.0).clamp(0, 255).to(torch.uint8).unsqueeze(0)

                # Get Depth Blending with GS background
                sim_depth = depth_data.squeeze(-1)  # Shape: (envs, H, W, 1) -> (envs, H, W)
                bg_depth = gs_result.depth.squeeze(-1)  # Shape: (envs, H, W, 1) -> (envs, H, W)
                if isinstance(bg_depth, np.ndarray):
                    bg_depth = torch.from_numpy(bg_depth)
                bg_depth = bg_depth.to(self.device)
                # Use torch.where for depth composition
                depth_comp = torch.where(foreground_mask > 0.5, sim_depth, bg_depth)
                depth_data = depth_comp.unsqueeze(0).unsqueeze(-1)

            camera_states[camera.name] = CameraState(
                rgb=rgb_data,
                depth=depth_data,
                instance_seg=instance_seg_data,
                instance_seg_id2label=instance_seg_id2label,
                instance_id_seg=instance_id_seg_data,
                instance_id_seg_id2label=instance_id_seg_id2label,
                pos=camera_inst.data.pos_w.clone(),
                quat_world=camera_inst.data.quat_w_world.clone(),
                intrinsics=torch.tensor(camera.intrinsics, device=self.device)[None, ...].repeat(self.num_envs, 1, 1),
            )
        extras = self.get_extra()
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    def _camera_info_mapping(self, camera_inst) -> dict:
        info = getattr(getattr(camera_inst, "data", None), "info", None)
        if isinstance(info, list):
            return info[0] if info else {}
        return info if isinstance(info, dict) else {}

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb
        from isaaclab.sim import SimulationContext

        if event.input == carb.input.KeyboardInput.V:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._render_viewport = not self._render_viewport

            if not self._render_viewport:
                if self.sim.has_rtx_sensors():
                    self.sim.set_render_mode(SimulationContext.RenderMode.PARTIAL_RENDERING)
                else:
                    self.sim.set_render_mode(SimulationContext.RenderMode.NO_RENDERING)
            else:
                self.sim.set_render_mode(SimulationContext.RenderMode.FULL_RENDERING)

    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        self._actions_cache = actions
        dict_action_batch: ActionBatch | None = actions if isinstance(actions, list) else None
        actions_tensor = (
            None if dict_action_batch is not None else action_input_to_tensor(self, actions, device=self.device)
        )
        if dict_action_batch is not None and len(dict_action_batch) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} dict actions, got {len(dict_action_batch)}.")

        offset = 0
        for i, robot in enumerate(self.robots):
            robot_inst = self.scene.articulations[robot.name]
            sorted_joint_names = self.get_joint_names(robot.name, sort=True)
            joint_count = len(sorted_joint_names)

            name_to_sorted_idx = {name: idx for idx, name in enumerate(sorted_joint_names)}

            joint_ids = []
            action_indices = []
            for joint_id, joint_name in enumerate(robot_inst.joint_names):
                if joint_name in name_to_sorted_idx:
                    joint_ids.append(joint_id)
                    action_indices.append(name_to_sorted_idx[joint_name])

            if not joint_ids:
                continue

            if dict_action_batch is not None:
                robot_targets_sorted = torch.zeros(
                    (self.num_envs, joint_count), dtype=torch.float32, device=self.device
                )
                for env_id, env_action in enumerate(dict_action_batch):
                    robot_action = env_action.get(robot.name) or {}
                    if self._manual_pd_on[i] and robot_action.get("dof_effort_target") is not None:
                        source_targets = robot_action["dof_effort_target"] or {}
                    else:
                        source_targets = robot_action.get("dof_pos_target") or {}
                    for joint_name, sorted_idx in name_to_sorted_idx.items():
                        if joint_name in source_targets:
                            robot_targets_sorted[env_id, sorted_idx] = float(source_targets[joint_name])
            else:
                if offset + joint_count > actions_tensor.shape[1]:
                    raise ValueError("Mismatch between provided actions and expected joint count.")
                robot_targets_sorted = actions_tensor[:, offset : offset + joint_count]
                offset += joint_count

            joint_targets = robot_targets_sorted[:, action_indices]

            if self._manual_pd_on[i]:
                # torque / effort control
                robot_inst.set_joint_effort_target(joint_targets, joint_ids=joint_ids)
            else:
                # position control
                robot_inst.set_joint_position_target(joint_targets, joint_ids=joint_ids)

            robot_inst.write_data_to_sim()

    def _simulate(self):
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.scene.write_data_to_sim()

        # Decimation: run physics multiple times per control step for better stability
        self._render_current = False
        for _ in range(self.decimation):
            self._physics_step_counter += 1
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            if self._physics_step_counter % self.render_interval == 0 and is_rendering:
                self.sim.render()

        # Force update kinematic objects to ensure visual mesh stays in sync
        for obj in self.objects:
            if obj.fix_base_link:
                if isinstance(obj, ArticulationObjCfg):
                    obj_inst = self.scene.articulations[obj.name]
                else:
                    obj_inst = self.scene.rigid_objects[obj.name]
                obj_inst.update(dt=0.0)

        # Ensure camera pose is correct, especially for the first few frames
        if self._physics_step_counter < 5:
            self._update_camera_pose()

    def _add_robot(self, robot: ArticulationObjCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        control_type = getattr(robot, "control_type", None)
        manual_pd = any(mode == "effort" for mode in control_type.values()) if control_type else False
        self._manual_pd_on.append(manual_pd)
        asset_path = resolve_isaacsim_file_path(robot)

        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=asset_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=not robot.enabled_gravity,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=getattr(
                    robot, "max_depenetration_velocity", self.scenario.sim_params.max_depenetration_velocity
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=robot.enabled_self_collisions,
                fix_root_link=robot.fix_base_link,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            collision_props=getattr(
                robot,
                "collision_props",
                sim_utils.CollisionPropertiesCfg(
                    contact_offset=getattr(robot, "contact_offset", self.scenario.sim_params.contact_offset),
                    rest_offset=getattr(robot, "rest_offset", self.scenario.sim_params.rest_offset),
                ),
            ),
        )
        cfg = ArticulationCfg(
            spawn=spawn_cfg,
            actuators={
                jn: ImplicitActuatorCfg(
                    joint_names_expr=[jn],
                    effort_limit_sim=actuator.effort_limit_sim,
                    velocity_limit_sim=actuator.velocity_limit_sim,
                    stiffness=actuator.stiffness if not manual_pd else 0.0,
                    damping=actuator.damping if not manual_pd else 0.0,
                    armature=actuator.armature if actuator.armature is not None else getattr(robot, "armature", 0.01),
                )
                for jn, actuator in robot.actuators.items()
            },
        )
        cfg.prim_path = f"/World/envs/env_.*/{robot.name}"
        init_state = ArticulationCfg.InitialStateCfg(
            pos=robot.default_position,
            rot=robot.default_orientation,
            joint_pos=robot.default_joint_positions,
            joint_vel={".*": 0.0},
        )
        cfg.init_state = init_state
        # NOTE `velocity_limit` here won't take effect
        for joint_name, actuator in robot.actuators.items():
            cfg.actuators[joint_name].velocity_limit = actuator.velocity_limit
        robot_inst = Articulation(cfg)
        self.scene.articulations[robot.name] = robot_inst

    def _add_object(self, obj: BaseObjCfg) -> None:
        """Add an object to the scene."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

        assert isinstance(obj, BaseObjCfg)
        prim_path = f"/World/envs/env_.*/{obj.name}"

        ## Articulation object
        if isinstance(obj, ArticulationObjCfg):
            asset_path = resolve_isaacsim_file_path(obj)
            articulation_cfg = ArticulationCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=asset_path,
                    scale=obj.scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=not obj.enabled_gravity),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=obj.fix_base_link),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=obj.default_position,
                    rot=obj.default_orientation,
                ),
                actuators={},
            )
            self.scene.articulations[obj.name] = Articulation(articulation_cfg)
            return

        if obj.fix_base_link:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                max_depenetration_velocity=getattr(
                    obj, "max_depenetration_velocity", self.scenario.sim_params.max_depenetration_velocity
                ),
            )
        else:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=not obj.enabled_gravity)
        if obj.collision_enabled:
            collision_props = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=getattr(obj, "contact_offset", self.scenario.sim_params.contact_offset),
                rest_offset=getattr(obj, "rest_offset", self.scenario.sim_params.rest_offset),
            )
        else:
            collision_props = None

        ## Primitive object
        if isinstance(obj, PrimitiveCubeCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCuboidCfg(
                        size=obj.size,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(float(c) for c in obj.color[:3])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=obj.default_position,
                        rot=obj.default_orientation,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveSphereCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshSphereCfg(
                        radius=obj.radius,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(float(c) for c in obj.color[:3])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=obj.default_position,
                        rot=obj.default_orientation,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveCylinderCfg):
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.MeshCylinderCfg(
                        radius=obj.radius,
                        height=obj.height,
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(float(c) for c in obj.color[:3])
                        ),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=obj.default_position,
                        rot=obj.default_orientation,
                    ),
                )
            )
            return
        if isinstance(obj, PrimitiveFrameCfg):
            usd_path = obj.usd_path or _primitive_frame_usd_path()
            self.scene.rigid_objects[obj.name] = RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=usd_path,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True, kinematic_enabled=True
                        ),  # fixed
                        collision_props=None,  # no collision
                        scale=obj.scale,
                    ),
                )
            )
            return

        ## Rigid object
        if isinstance(obj, RigidObjCfg):
            asset_path = resolve_isaacsim_file_path(obj)
            usd_file_cfg = sim_utils.UsdFileCfg(
                usd_path=asset_path,
                rigid_props=rigid_props,
                collision_props=collision_props,
                scale=obj.scale,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            )
            if isinstance(obj, RigidObjCfg):
                self.scene.rigid_objects[obj.name] = RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=usd_file_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=obj.default_position, rot=obj.default_orientation
                        ),
                    )
                )
                return

        raise ValueError(f"Unsupported object type: {type(obj)}")

    def _load_terrain(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab.terrains.trimesh import mesh_terrains_cfg as mesh_cfg

        # Auto-download terrain material if missing (same as DR)
        mdl_path = "roboverse_data/materials/arnold/Wood/Ash.mdl"
        if not os.path.exists(mdl_path):
            try:
                from metasim.utils.hf_util import check_and_download_single, extract_texture_paths_from_mdl

                log.info(f"Downloading terrain material: {mdl_path}")
                check_and_download_single(mdl_path)

                # Download textures (same as DR's apply_mdl_material)
                if os.path.exists(mdl_path):
                    try:
                        texture_paths = extract_texture_paths_from_mdl(mdl_path)
                        for tex_path in texture_paths:
                            if not os.path.exists(tex_path):
                                log.debug(f"Downloading texture: {tex_path}")
                                check_and_download_single(tex_path)
                    except Exception as e:
                        log.debug(f"Failed to download textures: {e}")
            except Exception as e:
                log.warning(f"Failed to download terrain material {mdl_path}: {e}")

        ground_padding = 8
        num_cols = math.ceil(math.sqrt(self._num_envs)) + ground_padding
        num_rows = num_cols
        # make each tile at least env_spacing (add a margin so robot never touches tile boundary)
        tile = 1.25 * self.scenario.env_spacing

        plane_gen_cfg = TerrainGeneratorCfg(
            size=(tile, tile),
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=None,
            use_cache=False,
            sub_terrains={
                "flat": mesh_cfg.MeshPlaneTerrainCfg(
                    proportion=1.0,
                ),
            },
        )

        ground_cfg = getattr(self.scenario, "ground", None)
        static_friction = getattr(ground_cfg, "static_friction", 1.0) if ground_cfg is not None else 1.0
        dynamic_friction = getattr(ground_cfg, "dynamic_friction", 1.0) if ground_cfg is not None else 1.0
        restitution = getattr(ground_cfg, "restitution", 0.0) if ground_cfg is not None else 0.0

        terrain_config = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=plane_gen_cfg,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
            debug_vis=False,
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=mdl_path,
                project_uvw=True,
                texture_scale=(1.0, 1.0),
                albedo_brightness=1.2,
            ),
        )

        terrain_config.num_envs = self.scene.cfg.num_envs
        terrain_config.env_spacing = self.scene.cfg.env_spacing
        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins
        if ground_cfg is not None:
            self._build_custom_terrain_mesh(ground_cfg)

    def _build_custom_terrain_mesh(self, ground_cfg) -> None:
        """Procedurally author a USD mesh using TerrainGenerator."""
        tg = TerrainGenerator(ground_cfg)
        stage_vertices, stage_triangles, height_mat = tg.generate_terrain(ground_cfg, type="both")

        max_triangles = getattr(ground_cfg, "max_mesh_triangles", 20000)
        raw_heights = (height_mat / tg.vertical_scale).astype(np.float32)
        ds_heights, scale_x, scale_y = self._downsample_height_field(raw_heights, tg.horizontal_scale, max_triangles)
        if not math.isclose(scale_x, scale_y, rel_tol=1e-4):
            stage_vertices = stage_vertices.copy()
            stage_vertices[:, 1] *= scale_y / max(scale_x, 1e-6)

        stage_height_mat = ds_heights * tg.vertical_scale
        if stage_triangles.shape[0] != stage_triangles.shape[0]:
            log.info(
                "Downsampled IsaacSim terrain mesh from %d to %d triangles (limit=%d).",
                len(stage_triangles),
                len(stage_triangles),
                max_triangles,
            )

        self._ground_mesh_vertices = stage_vertices
        self._ground_mesh_triangles = stage_triangles.astype(np.int32)
        self._height_mat = stage_height_mat

        # Center the terrain at the origin
        terrain_vertices = self._ground_mesh_vertices.copy()
        half_width = (terrain_vertices[:, 0].max() - terrain_vertices[:, 0].min()) / 2.0
        half_height = (terrain_vertices[:, 1].max() - terrain_vertices[:, 1].min()) / 2.0
        terrain_vertices[:, 0] -= half_width
        terrain_vertices[:, 1] -= half_height

        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        try:
            import omni.isaac.core.utils.prims as prim_utils
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils

        stage = prim_utils.get_current_stage()
        if stage is None:
            log.error("IsaacSim stage is not available; cannot create terrain mesh.")
            return

        ground_root_path = "/World/ground"
        ground_root = stage.GetPrimAtPath(ground_root_path)
        if not ground_root or not ground_root.IsValid():
            ground_root = stage.DefinePrim(ground_root_path, "Xform")
        else:
            for child in list(ground_root.GetChildren()):
                stage.RemovePrim(child.GetPath())

        mesh_path = f"{ground_root_path}/generated_mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in terrain_vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(self._ground_mesh_triangles))
        mesh.CreateFaceVertexIndicesAttr(self._ground_mesh_triangles.flatten().tolist())
        mesh.CreateExtentAttr(self._compute_mesh_extent(terrain_vertices))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.6, 0.6)])

        # Enable physics collisions on the generated mesh
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision_api.CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        rigid_api.CreateRigidBodyEnabledAttr(False)

        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        physx_collision_api.CreateRestOffsetAttr(0.0)
        physx_collision_api.CreateContactOffsetAttr(0.02)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh.GetPrim())

        static_friction = getattr(ground_cfg, "static_friction", 1.0)
        dynamic_friction = getattr(ground_cfg, "dynamic_friction", 1.0)
        restitution = getattr(ground_cfg, "restitution", 0.0)

        material_path = "/World/Materials/TerrainMaterial"
        usd_material = UsdShade.Material.Define(stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(usd_material.GetPrim())
        material_api.CreateStaticFrictionAttr(static_friction)
        material_api.CreateDynamicFrictionAttr(dynamic_friction)
        material_api.CreateRestitutionAttr(restitution)
        mat_binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
        mat_binding.Bind(usd_material, materialPurpose="physics")

        self._ground_mesh_vertices = terrain_vertices
        self._terrain_margin = tg.margin

        log.info(
            "Generated IsaacSim terrain mesh with %d vertices and %d triangles.",
            len(self._ground_mesh_vertices),
            len(self._ground_mesh_triangles),
        )

    def _downsample_height_field(
        self, height_field_raw: np.ndarray, horizontal_scale: float, max_triangles: int
    ) -> tuple[np.ndarray, float, float]:
        """Reduce height-field resolution to keep triangle count manageable for PhysX GPU buffers."""
        rows, cols = height_field_raw.shape
        total_triangles = 2 * max(rows - 1, 0) * max(cols - 1, 0)
        if max_triangles is None or max_triangles <= 0 or total_triangles <= max_triangles or total_triangles == 0:
            return height_field_raw, horizontal_scale, horizontal_scale

        reduction = math.sqrt(total_triangles / max_triangles)
        new_rows = max(2, math.floor((rows - 1) / reduction) + 1)
        new_cols = max(2, math.floor((cols - 1) / reduction) + 1)

        src_x = np.linspace(0.0, rows - 1, rows, dtype=np.float32)
        src_y = np.linspace(0.0, cols - 1, cols, dtype=np.float32)
        interpolator = RegularGridInterpolator((src_x, src_y), height_field_raw, bounds_error=False, fill_value=None)
        dst_x = np.linspace(0.0, rows - 1, new_rows, dtype=np.float32)
        dst_y = np.linspace(0.0, cols - 1, new_cols, dtype=np.float32)
        grid = np.stack(np.meshgrid(dst_x, dst_y, indexing="ij"), axis=-1)
        downsampled = interpolator(grid)

        total_width_x = (rows - 1) * horizontal_scale
        total_width_y = (cols - 1) * horizontal_scale
        scale_x = total_width_x / max(new_rows - 1, 1)
        scale_y = total_width_y / max(new_cols - 1, 1)

        return downsampled.astype(np.float32), scale_x, scale_y

    @staticmethod
    def _compute_mesh_extent(vertices: np.ndarray):
        from pxr import Gf

        min_corner = vertices.min(axis=0)
        max_corner = vertices.max(axis=0)
        return [Gf.Vec3f(*min_corner.tolist()), Gf.Vec3f(*max_corner.tolist())]

    def _load_scene(self) -> None:
        """Load scene from SceneCfg configuration.

        Loads USD scene files into each environment if scene configuration is provided.
        Supports position, rotation, and scale transformations.
        """
        if self.scenario.scene is None:
            return

        scene_cfg = self.scenario.scene

        # Only support USD path for now
        if scene_cfg.usd_path is None:
            log.warning("Scene USD path is None, skipping scene loading")
            return

        try:
            import omni.isaac.core.utils.prims as prim_utils
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils

        from pxr import Gf, UsdGeom

        # Get current stage
        stage = prim_utils.get_current_stage()
        if not stage:
            log.error("Failed to get current stage")
            return

        # Get scene name, default to "scene"
        scene_name = scene_cfg.name if scene_cfg.name else "scene"

        # Determine scene path pattern for all environments
        scene_prim_path = f"/World/envs/env_.*/{scene_name}"

        # Get absolute path
        usd_path = os.path.abspath(scene_cfg.usd_path)
        if not os.path.exists(usd_path):
            log.error(f"Scene USD file not found: {usd_path}")
            return

        # Load scene for source environment (env_0)
        source_scene_path = f"/World/envs/env_0/{scene_name}"

        # Add USD reference to stage
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage

            add_reference_to_stage(usd_path, source_scene_path)
        except ImportError:
            # Fallback: use USD API directly
            ref_prim = stage.DefinePrim(source_scene_path, "Xform")
            if not ref_prim:
                log.error(f"Failed to create prim at {source_scene_path}")
                return
            ref_prim.GetReferences().AddReference(usd_path)

        # Apply transformations if specified
        scene_prim = stage.GetPrimAtPath(source_scene_path)
        if scene_prim.IsValid():
            xformable = UsdGeom.Xformable(scene_prim)

            # Clear existing transform operations
            xformable.ClearXformOpOrder()

            # Apply scale if specified
            if scene_cfg.scale is not None:
                scale_op = xformable.AddScaleOp()
                scale_op.Set(Gf.Vec3d(*scene_cfg.scale))

            # Apply rotation if specified (using quaternion directly)
            if scene_cfg.quat is not None:
                # SceneCfg quat format is (w, x, y, z)
                qw, qx, qy, qz = scene_cfg.quat
                # USD quaternion format is (real, imag_i, imag_j, imag_k) = (w, x, y, z)
                # Use Quatf (float) instead of Quatd (double) as USD expects float precision
                quat_gf = Gf.Quatf(float(qw), float(qx), float(qy), float(qz))
                # Use orient op to set quaternion rotation directly
                orient_op = xformable.AddOrientOp()
                orient_op.Set(quat_gf)

            # Apply fixed position offset if specified
            if scene_cfg.default_position is not None:
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(*scene_cfg.default_position))
                log.debug(f"Set scene position offset: {scene_cfg.default_position}")

            log.info(f"Loaded scene from {usd_path} at {source_scene_path}")

    def _load_render_settings(self) -> None:
        try:
            import carb
            import omni.replicator.core as rep
        except ImportError:
            # Physics-only / no-camera headless runs launch a minimal Kit app that does
            # not enable the replicator (rendering) extension. Render settings only affect
            # image quality, so skip them rather than hard-crash when no images are needed.
            log.warning("omni.replicator unavailable; skipping render settings (no-camera headless run).")
            return

        # from omni.rtx.settings.core.widgets.pt_widgets import PathTracingSettingsFrame

        rep.settings.set_render_rtx_realtime()  # fix noising rendered images

        settings = carb.settings.get_settings()
        if self.scenario.render.mode == "pathtracing":
            settings.set_string("/rtx/rendermode", "PathTracing")
        elif self.scenario.render.mode == "realtime_pathtracing":
            # RTX - Real-Time 2.0: the renderer was registered at Kit boot
            # (rt2 enable flags injected in _init_scene when this handler owns
            # the app), so this runtime write actually engages it; without that
            # registration the write is silently refused. The check below
            # verifies engagement either way.
            settings.set_string("/rtx/rendermode", "RealTimePathTracing")
        elif self.scenario.render.mode == "raytracing":
            settings.set_string("/rtx/rendermode", "RaytracedLighting")
        elif self.scenario.render.mode == "rasterization":
            raise ValueError("Isaaclab does not support rasterization")
        else:
            raise ValueError(f"Unknown render mode: {self.scenario.render.mode}")

        if self.scenario.render.mode == "realtime_pathtracing":
            applied = settings.get_as_string("/rtx/rendermode")
            if applied != "RealTimePathTracing":
                if self._owns_simulation_app:
                    hint = (
                        "The rt2 registration flags were injected at Kit boot, so this Isaac Sim build "
                        "most likely lacks RTX - Real-Time 2.0; use render.mode='raytracing' or "
                        "'pathtracing'."
                    )
                else:
                    hint = (
                        "This handler received an already-running SimulationApp, and the RealTimePathTracing "
                        "renderer must be REGISTERED at Kit boot — launch your app with "
                        "--/persistent/rtx/modes/rt2/enabled=true --/rtx-transient/rt2Enabled=true in its "
                        "Kit argv (without registration, /rtx/rendermode writes are silently refused)."
                    )
                raise RuntimeError(
                    f"RTX - Real-Time 2.0 (RealTimePathTracing) is not engaged (got /rtx/rendermode={applied!r}). "
                    + hint
                )

        log.info(f"Render mode: {settings.get_as_string('/rtx/rendermode')}")
        log.info(f"Render totalSpp: {settings.get('/rtx/pathtracing/totalSpp')}")
        log.info(f"Render spp: {settings.get('/rtx/pathtracing/spp')}")
        log.info(f"Render adaptiveSampling/enabled: {settings.get('/rtx/pathtracing/adaptiveSampling/enabled')}")
        log.info(f"Render maxBounces: {settings.get('/rtx/pathtracing/maxBounces')}")

    def _load_sensors(self) -> None:
        # Contact sensing is initialized lazily via `init_contact_sensor()` so
        # scenarios that never query contact forces do not pay the extra PhysX
        # memory cost.
        return

    def init_contact_sensor(self, robot_name: str) -> None:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        if self.contact_sensor is not None:
            return
        if self.scene is None:
            raise RuntimeError("Scene is not initialized. Launch the handler before creating a contact sensor.")

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{robot_name}/.*",
            history_length=3,
            update_period=0.005,
            force_threshold=10.0,
            track_air_time=True,
        )
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if hasattr(self.contact_sensor, "_initialize_callback"):
            self.contact_sensor._initialize_callback(None)
        self.contact_sensor.update(dt=0.0)

    def _load_lights(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        from metasim.scenario.lights import (
            CylinderLightCfg,
            DiskLightCfg,
            DistantLightCfg,
            DomeLightCfg,
            SphereLightCfg,
        )

        # Use lights from scenario configuration if available
        if hasattr(self.scenario, "lights") and self.scenario.lights:
            for i, light_cfg in enumerate(self.scenario.lights):
                if isinstance(light_cfg, DistantLightCfg):
                    self._add_distant_light(light_cfg, i)
                elif isinstance(light_cfg, CylinderLightCfg):
                    self._add_cylinder_light(light_cfg, i)
                elif isinstance(light_cfg, DomeLightCfg):
                    self._add_dome_light(light_cfg, i)
                elif isinstance(light_cfg, SphereLightCfg):
                    self._add_sphere_light(light_cfg, i)
                elif isinstance(light_cfg, DiskLightCfg):
                    self._add_disk_light(light_cfg, i)
                else:
                    log.warning(f"Unsupported light type: {type(light_cfg)}, skipping...")
        else:
            # Fallback to default light if no lights are configured
            log.info("No lights configured, using default distant light")
            spawn_light(
                "/World/DefaultLight",
                sim_utils.DistantLightCfg(intensity=2000.0, angle=0.53),  # Increased default intensity
                orientation=(1.0, 0.0, 0.0, 0.0),
                translation=(0, 0, 10),
            )

    def _add_distant_light(self, light_cfg, light_index: int) -> None:
        """Add a distant light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Use configured name if available, otherwise fall back to index-based naming
        light_name = (
            f"/World/{light_cfg.name}"
            if hasattr(light_cfg, "name") and light_cfg.name and light_cfg.name != "light"
            else f"/World/DistantLight_{light_index}"
        )

        # Create Isaac Lab distant light configuration
        isaac_light_cfg = sim_utils.DistantLightCfg(
            intensity=light_cfg.intensity,
            angle=0.53,  # Default angle, could be made configurable
            color=light_cfg.color,
        )

        # Use the quaternion from light configuration
        orientation = light_cfg.quat

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=orientation,
            translation=(0, 0, 10),  # Distant lights don't need specific translation
        )

        log.debug(
            f"Added distant light {light_name} with intensity {light_cfg.intensity}, "
            f"polar={light_cfg.polar}°, azimuth={light_cfg.azimuth}°"
        )

    def _add_cylinder_light(self, light_cfg, light_index: int) -> None:
        """Add a cylinder light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Use configured name if available, otherwise fall back to index-based naming
        light_name = (
            f"/World/{light_cfg.name}"
            if hasattr(light_cfg, "name") and light_cfg.name and light_cfg.name != "light"
            else f"/World/CylinderLight_{light_index}"
        )

        # Create Isaac Lab cylinder light configuration
        isaac_light_cfg = sim_utils.CylinderLightCfg(
            intensity=light_cfg.intensity, radius=light_cfg.radius, length=light_cfg.length, color=light_cfg.color
        )

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=light_cfg.rot,
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added cylinder light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius}, length={light_cfg.length}"
        )

    def _add_dome_light(self, light_cfg, light_index: int) -> None:
        """Add a dome light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Use configured name if available, otherwise fall back to index-based naming
        light_name = (
            f"/World/{light_cfg.name}"
            if hasattr(light_cfg, "name") and light_cfg.name and light_cfg.name != "light"
            else f"/World/DomeLight_{light_index}"
        )

        # Create Isaac Lab dome light configuration
        isaac_light_cfg = sim_utils.DomeLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
        )

        # Add texture if specified
        if light_cfg.texture_file is not None:
            isaac_light_cfg.texture_file = light_cfg.texture_file

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=(1.0, 0.0, 0.0, 0.0),
            translation=(0, 0, 0),  # Dome lights are typically at origin
        )

        log.debug(f"Added dome light {light_name} with intensity {light_cfg.intensity}")

    def _add_sphere_light(self, light_cfg, light_index: int) -> None:
        """Add a sphere light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Use configured name if available, otherwise fall back to index-based naming
        light_name = (
            f"/World/{light_cfg.name}"
            if hasattr(light_cfg, "name") and light_cfg.name and light_cfg.name != "light"
            else f"/World/SphereLight_{light_index}"
        )

        # Create Isaac Lab sphere light configuration
        isaac_light_cfg = sim_utils.SphereLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
            radius=light_cfg.radius,
            normalize=light_cfg.normalize,
        )

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=(1.0, 0.0, 0.0, 0.0),
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added sphere light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius} at {light_cfg.pos}"
        )

    def _add_disk_light(self, light_cfg, light_index: int) -> None:
        """Add a disk light to the scene based on configuration."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

        # Use configured name if available, otherwise fall back to index-based naming
        light_name = (
            f"/World/{light_cfg.name}"
            if hasattr(light_cfg, "name") and light_cfg.name and light_cfg.name != "light"
            else f"/World/DiskLight_{light_index}"
        )

        # Create Isaac Lab disk light configuration
        isaac_light_cfg = sim_utils.DiskLightCfg(
            intensity=light_cfg.intensity,
            color=light_cfg.color,
            radius=light_cfg.radius,
            normalize=light_cfg.normalize,
        )

        spawn_light(
            light_name,
            isaac_light_cfg,
            orientation=light_cfg.rot,
            translation=light_cfg.pos,
        )

        log.debug(
            f"Added disk light {light_name} with intensity {light_cfg.intensity}, "
            f"radius={light_cfg.radius} at {light_cfg.pos}"
        )

    # def _load_ground(self) -> None:
    #     import isaaclab.sim as sim_utils
    #     cfg_ground = sim_utils.GroundPlaneCfg(
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    #         color=(1.0,1.0,1.0),
    #     )
    #     cfg_ground.func("/World/ground", cfg_ground)
    # import isaacsim.core.experimental.utils.prim as prim_utils
    # import omni
    # from pxr import Sdf, UsdShade
    # ground_prim = prim_utils.get_prim_at_path("/World/ground")
    # material = UsdShade.MaterialBindingAPI(ground_prim).GetDirectBinding().GetMaterial()
    # shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))
    # # Correspond to Shader -> Inputs -> UV -> Texture Tiling (in Isaac Sim 4.2.0)
    # shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set((10,10))

    def _get_pose(
        self, obj_name: str, obj_subpath: str | None = None, env_ids: list[int] | None = None
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if obj_name in self.scene.rigid_objects:
            obj_inst = self.scene.rigid_objects[obj_name]
        elif obj_name in self.scene.articulations:
            obj_inst = self.scene.articulations[obj_name]
        else:
            raise ValueError(f"Object {obj_name} not found")

        if obj_subpath is None:
            pos = obj_inst.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
            rot = obj_inst.data.root_quat_w[env_ids]
        else:
            log.error(f"Subpath {obj_subpath} is not supported in IsaacsimHandler.get_pose")

        assert pos.shape == (len(env_ids), 3)
        assert rot.shape == (len(env_ids), 4)
        return pos, rot

    @property
    def device(self) -> torch.device:
        return self._device

    def _set_object_pose(
        self,
        object: BaseObjCfg,
        position: torch.Tensor,  # (num_envs, 3)
        rotation: torch.Tensor,  # (num_envs, 4)
        env_ids: list[int] | None = None,
    ) -> None:
        """
        Set the pose of an object, set the velocity to zero
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert position.shape == (len(env_ids), 3)
        assert rotation.shape == (len(env_ids), 4)

        if isinstance(object, BaseArticulationObjCfg):
            obj_inst = self.scene.articulations[object.name]
        elif isinstance(object, BaseRigidObjCfg):
            obj_inst = self.scene.rigid_objects[object.name]
        else:
            raise ValueError(f"Invalid object type: {type(object)}")

        pose = torch.concat(
            [
                position.to(self.device, dtype=torch.float32) + self.scene.env_origins[env_ids],
                rotation.to(self.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        obj_inst.write_root_pose_to_sim(pose, env_ids=torch.tensor(env_ids, device=self.device))
        obj_inst.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device, dtype=torch.float32),
            env_ids=torch.tensor(env_ids, device=self.device),
        )  # ! critical
        obj_inst.write_data_to_sim()

        if object.fix_base_link:
            obj_inst.update(dt=0.0)

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = deepcopy(self.scene.articulations[obj_name].joint_names)
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _set_object_joint_pos(
        self,
        object: BaseObjCfg,
        joint_pos: torch.Tensor,  # (num_envs, num_joints)
        env_ids: list[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        assert joint_pos.shape[0] == len(env_ids)
        pos = joint_pos.to(self.device)
        vel = torch.zeros_like(pos)
        obj_inst = self.scene.articulations[object.name]
        obj_inst.write_joint_state_to_sim(pos, vel, env_ids=torch.tensor(env_ids, device=self.device))
        obj_inst.write_data_to_sim()

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            body_names = deepcopy(self.scene.articulations[obj_name].body_names)
            if sort:
                body_names.sort()
            return body_names
        else:
            return []

    def _add_pinhole_camera(self, camera: PinholeCameraCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        data_type_map = {
            "rgb": "rgb",
            "depth": "depth",
            "instance_seg": "instance_segmentation_fast",
            "instance_id_seg": "instance_id_segmentation_fast",
        }
        if camera.mount_to is None:
            prim_path = f"/World/envs/env_.*/{camera.name}"
            # Use default offset, will be set by set_world_poses_from_view later
            offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
        else:
            prim_path = f"/World/envs/env_.*/{camera.mount_to}/{camera.mount_link}/{camera.name}"
            offset = CameraCfg.OffsetCfg(pos=camera.mount_pos, rot=camera.mount_quat, convention="world")

        camera_inst = Camera(
            CameraCfg(
                prim_path=prim_path,
                offset=offset,
                data_types=[data_type_map[dt] for dt in camera.data_types],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=camera.focal_length,
                    focus_distance=camera.focus_distance,
                    horizontal_aperture=camera.horizontal_aperture,
                    clipping_range=camera.clipping_range,
                ),
                width=camera.width,
                height=camera.height,
                colorize_instance_segmentation=False,
                colorize_instance_id_segmentation=False,
            )
        )
        self.scene.sensors[camera.name] = camera_inst
        log.debug(f"Added camera {camera.name} to scene with prim_path: {prim_path}")

    def refresh_render(self, passes: int = 2) -> None:
        """Render the current state: ``passes`` x (scene update, ``sim.render()``, sensor update).

        Two passes are the safe default after teleporting prims (the first pass propagates the
        transforms, the second sees them). A caller that already flushed the writes and renders
        every step (``HybridSimHandler``) uses one. Skipped while ``_defer_all_visual_flushes``
        is set, so a batch of writes renders once at the end instead of once per write.
        """
        if getattr(self, "_defer_all_visual_flushes", False):
            return
        physics_dt = float(getattr(self, "physics_dt", 0.0))
        if self.scene is not None:
            try:
                self.scene.update(dt=0.0)
            except Exception as err:
                log.debug(f"Scene update failed during visual refresh: {err}")
        sensors = getattr(self.scene, "sensors", {}) if self.scene is not None else {}
        for i in range(max(1, int(passes))):
            if self.sim is not None:
                try:
                    self.sim.render()
                except Exception as err:
                    log.debug(f"SimulationContext render failed during visual refresh pass {i + 1}: {err}")
            for sensor in sensors.values():
                update = getattr(sensor, "update", None)
                if callable(update):
                    try:
                        update(dt=physics_dt)
                    except Exception as err:
                        log.debug(f"Sensor update failed during visual refresh pass {i + 1}: {err}")
        self._render_current = True

    def flush_visual_updates(self, *, wait_for_materials: bool = False, settle_passes: int = 2) -> None:
        """Drive SimulationApp/scene/sensors for a few frames to settle visual state.

        Global defer mechanism: If _defer_all_visual_flushes is True, skip flush entirely.
        This enables atomic batch randomization without intermediate rendering overhead.
        """
        # Check global defer flag (for batch randomization)
        if getattr(self, "_defer_all_visual_flushes", False):
            return  # Skip flush, will be done by batch controller

        passes = max(1, settle_passes)
        sim_app = getattr(self, "simulation_app", None)
        reason = "material refresh" if wait_for_materials else "visual flush"

        for _ in range(passes):
            if sim_app is not None:
                try:
                    sim_app.update()
                except Exception as err:
                    log.debug(f"SimulationApp update failed during {reason}: {err}")

            if self.scene is not None:
                try:
                    self.scene.update(dt=0)
                except Exception as err:
                    log.debug(f"Scene update failed during {reason}: {err}")

            if self.sim is not None:
                try:
                    if self.sim.has_gui() or self.sim.has_rtx_sensors():
                        self.sim.render()
                except Exception as err:
                    log.debug(f"Sim render failed during {reason}: {err}")

            sensors = getattr(self.scene, "sensors", {}) if self.scene is not None else {}
            for name, sensor in sensors.items():
                try:
                    sensor.update(dt=0)
                except Exception as err:
                    log.debug(f"Sensor {name} update failed during {reason}: {err}")

        if wait_for_materials:
            self._refresh_raytracing_acceleration()

    def _refresh_raytracing_acceleration(self) -> None:
        """Work around Isaac Sim 4.5 RTX BVH getting stale after material edits."""
        render_cfg = getattr(self.scenario, "render", None)
        if render_cfg is None or getattr(render_cfg, "mode", None) not in ("raytracing", "realtime_pathtracing"):
            return

        try:
            import carb
            import omni.kit.app
        except ImportError:
            return

        settings = carb.settings.get_settings()
        app = omni.kit.app.get_app()
        if settings is None or app is None:
            return

        enabled_path = "/rtx/raytracing/enabled"
        try:
            current_state = settings.get(enabled_path)
        except Exception as err:
            log.debug(f"Unable to read RTX setting {enabled_path}: {err}")
            current_state = None

        if current_state is None:
            current_state = True
            try:
                settings.set(enabled_path, current_state)
                app.update()
            except Exception as err:
                log.debug(f"Failed to initialize RTX setting {enabled_path}: {err}")
                return

        log.debug("Refreshing RTX acceleration structure after material update")
        try:
            settings.set(enabled_path, False)
            app.update()
            settings.set(enabled_path, current_state)
            app.update()

            gc_path = "/rtx/hydra/triggerGarbageCollection"
            settings.set(gc_path, True)
            app.update()
            settings.set(gc_path, False)
            app.update()

            if self.sim is not None:
                try:
                    self.sim.render()
                except Exception as err:
                    log.debug(f"Sim render during RTX refresh failed: {err}")
            if self.scene is not None:
                try:
                    self.scene.update(dt=0)
                except Exception as err:
                    log.debug(f"Scene update during RTX refresh failed: {err}")
        except Exception as err:
            log.debug(f"Failed to refresh RTX acceleration structure: {err}")

    def _get_camera_params(self, camera, camera_inst):
        """Get camera intrinsics and extrinsics for GS rendering.

        Compare IsaacSim camera pose vs look-at construction to find the correct transformation.

        Args:
            camera: PinholeCameraCfg object
            camera_inst: IsaacSim camera instance

        Returns:
            Ks_t: (3, 3) intrinsic matrix as torch tensor on device
            c2w_t: (4, 4) camera-to-world transformation matrix as torch tensor on device
        """
        # Get intrinsics

        Ks = np.array(camera.intrinsics, dtype=np.float32)
        Ks_t = torch.from_numpy(Ks).to(self.device)

        # # Method 1: Read from IsaacSim camera instance
        # p_isaac = camera_inst.data.pos_w[0].detach()  # Keep as tensor
        # q_wxyz = camera_inst.data.quat_w_world[0].detach()  # (w, x, y, z)

        # # Convert quaternion to rotation matrix using torch
        # # quaternion [w, x, y, z] -> rotation matrix
        # w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        # R_isaac = torch.stack([
        #     torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        #     torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        #     torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        # ]).to(self.device)

        # c2w_isaac = torch.eye(4, dtype=torch.float32, device=self.device)
        # c2w_isaac[:3, :3] = R_isaac
        # c2w_isaac[:3, 3] = p_isaac

        # Method 2: Build from look-at with -Z forward (OpenGL/MUJOCO convention)
        pos = torch.tensor(camera.pos, dtype=torch.float32, device=self.device)
        look = torch.tensor(camera.look_at, dtype=torch.float32, device=self.device)
        forward = look - pos
        forward = forward / torch.norm(forward)
        up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        right = torch.cross(forward, up_world)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)

        R_lookat = torch.stack([right, up, forward], dim=1)
        # Negate Z for -Z forward convention
        R_lookat[:, 2] = -R_lookat[:, 2]

        c2w_lookat = torch.eye(4, dtype=torch.float32, device=self.device)
        c2w_lookat[:3, :3] = R_lookat
        c2w_lookat[:3, 3] = pos

        # IsaacSim camera poses may not be reliable until after full scene updates.
        # Using look-at construction directly is more stable.
        return Ks_t, c2w_lookat
