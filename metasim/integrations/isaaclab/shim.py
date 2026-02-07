from __future__ import annotations

"""IsaacLab API shim.

This project supports running "IsaacLab manager-based" task code on MetaSim handlers
across multiple simulators. In practice, many task codebases import `isaaclab.*` for
config/MDP utilities even when they do not require Isaac Sim at runtime.

However, the real IsaacLab package may be unavailable in lightweight environments
(e.g., missing USD/pxr bindings). This shim provides a small, **pure-Python** subset
of the IsaacLab API so those tasks can still be imported and executed on non-Isaac
backends.

Important:
- This shim is **not** a simulator and does not provide IsaacSim/Omniverse features.
- MetaSim's `isaacsim` backend still requires the real IsaacLab/IsaacSim stack.
"""

import sys
import types
from dataclasses import MISSING, dataclass
from typing import Any, Callable, Literal

import torch

from metasim.utils.configclass import configclass
from metasim.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)


def _new_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__all__ = []
    return mod


def ensure_isaaclab_shim() -> bool:
    """Ensure an importable `isaaclab` API exists.

    Returns:
        True if the shim was installed, False if the real IsaacLab import worked.
    """
    try:
        import isaaclab

        # A top-level `import isaaclab` can succeed even when key submodules fail due to
        # missing native dependencies (e.g., USD/pxr). Probe the common import surface
        # used by task configs and MDP utilities.
        try:
            import isaaclab.envs.mdp
            import isaaclab.managers
            import isaaclab.utils  # noqa: F401
        except Exception:
            _install_shim()
            return True
        return False
    except Exception:
        # Real IsaacLab isn't importable. Install shim.
        _install_shim()
        return True


def _install_shim() -> None:
    # If a partial import exists, clear it out to avoid mixing shim+real modules.
    for key in list(sys.modules.keys()):
        if key == "isaaclab" or key.startswith("isaaclab."):
            sys.modules.pop(key, None)

    # Root package
    isaaclab = _new_module("isaaclab")
    sys.modules["isaaclab"] = isaaclab

    # ------------------------------------------------------------------
    # isaaclab.utils
    # ------------------------------------------------------------------
    utils = _new_module("isaaclab.utils")
    utils.configclass = configclass
    sys.modules["isaaclab.utils"] = utils
    isaaclab.utils = utils

    utils_configclass = _new_module("isaaclab.utils.configclass")
    utils_configclass.configclass = configclass
    sys.modules["isaaclab.utils.configclass"] = utils_configclass
    utils.configclass = configclass

    utils_noise = _new_module("isaaclab.utils.noise")

    @configclass
    class AdditiveUniformNoiseCfg:
        n_min: float = 0.0
        n_max: float = 0.0

    utils_noise.AdditiveUniformNoiseCfg = AdditiveUniformNoiseCfg
    sys.modules["isaaclab.utils.noise"] = utils_noise
    utils.noise = utils_noise

    utils_math = _new_module("isaaclab.utils.math")
    utils_math.matrix_from_quat = matrix_from_quat
    utils_math.subtract_frame_transforms = subtract_frame_transforms
    utils_math.quat_apply = quat_apply
    utils_math.quat_apply_inverse = quat_apply_inverse
    utils_math.quat_error_magnitude = quat_error_magnitude
    utils_math.quat_from_euler_xyz = quat_from_euler_xyz
    utils_math.quat_inv = quat_inv
    utils_math.quat_mul = quat_mul
    utils_math.sample_uniform = sample_uniform
    utils_math.yaw_quat = yaw_quat
    sys.modules["isaaclab.utils.math"] = utils_math
    utils.math = utils_math

    # Some IsaacLab-based task code imports small runtime helpers from isaaclab.utils.
    # Provide minimal pure-Python versions here so those configs can be imported and
    # (optionally) executed on non-Isaac simulators.

    class DelayBuffer:
        """Circular delay buffer for per-env tensors.

        This is a minimal implementation used by some task-side actuator models.
        It stores the last `max_delay + 1` values and returns values delayed by a
        per-environment integer time lag (in physics steps).
        """

        def __init__(self, max_delay: int, num_envs: int, *, device: str | torch.device | None = None):
            self.max_delay = int(max_delay)
            self.num_envs = int(num_envs)
            self.device = torch.device(device) if device is not None else torch.device("cpu")

            self._len = self.max_delay + 1
            self._head = 0
            self._time_lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._buf: torch.Tensor | None = None

        def _ensure_buf(self, value: torch.Tensor) -> torch.Tensor:
            if self._buf is not None:
                return self._buf
            shape = (self._len, self.num_envs) + tuple(value.shape[1:])
            self._buf = torch.zeros(shape, dtype=value.dtype, device=value.device)
            return self._buf

        def set_time_lag(self, time_lag: torch.Tensor, env_ids=None) -> None:
            lag = time_lag.to(device=self.device, dtype=torch.long)
            if env_ids is None or env_ids == slice(None):
                self._time_lag[:] = lag
                return
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._time_lag[ids] = lag

        def reset(self, env_ids=None) -> None:
            if self._buf is None:
                return
            if env_ids is None or env_ids == slice(None):
                self._buf.zero_()
                return
            ids = torch.as_tensor(env_ids, device=self._buf.device, dtype=torch.long)
            self._buf[:, ids] = 0

        def compute(self, value: torch.Tensor) -> torch.Tensor:
            if value.shape[0] != self.num_envs:
                raise ValueError(f"DelayBuffer.compute expects leading dim {self.num_envs}, got {value.shape[0]}.")

            buf = self._ensure_buf(value)
            buf[self._head] = value

            # Gather delayed indices per env: (head - lag) mod len.
            idx = (self._head - self._time_lag) % self._len
            out = buf[idx, torch.arange(self.num_envs, device=value.device)]

            self._head = (self._head + 1) % self._len
            return out

    utils.DelayBuffer = DelayBuffer

    utils_types = _new_module("isaaclab.utils.types")

    @dataclass
    class ArticulationActions:
        """Container for joint-space actions (subset used by task-side actuator code)."""

        joint_positions: torch.Tensor | None = None
        joint_velocities: torch.Tensor | None = None
        joint_efforts: torch.Tensor | None = None

    utils_types.ArticulationActions = ArticulationActions
    sys.modules["isaaclab.utils.types"] = utils_types
    utils.types = utils_types

    # ------------------------------------------------------------------
    # isaaclab.managers
    # ------------------------------------------------------------------
    managers = _new_module("isaaclab.managers")

    @configclass
    class SceneEntityCfg:
        name: str = MISSING
        body_names: str | list[str] | None = None
        joint_names: str | list[str] | None = None
        # IsaacLab typically "resolves" these at runtime (patterns -> indices). We include them
        # so IsaacLab-style term functions that reference `*_ids` can run unchanged.
        body_ids: slice | list[int] = slice(None)
        joint_ids: slice | list[int] = slice(None)

    @configclass
    class ObservationTermCfg:
        func: Callable[..., torch.Tensor] = MISSING
        params: dict[str, Any] = {}
        noise: Any | None = None

    @configclass
    class ObservationGroupCfg:
        enable_corruption: bool = False
        concatenate_terms: bool = True

    @configclass
    class RewardTermCfg:
        func: Callable[..., torch.Tensor] = MISSING
        weight: float = 1.0
        params: dict[str, Any] = {}

    @configclass
    class TerminationTermCfg:
        func: Callable[..., torch.Tensor] = MISSING
        time_out: bool = False
        params: dict[str, Any] = {}

    @configclass
    class EventTermCfg:
        func: Callable[..., Any] = MISSING
        mode: str = "startup"
        params: dict[str, Any] = {}
        interval_range_s: tuple[float, float] | None = None

    @configclass
    class CommandTermCfg:
        class_type: type = MISSING
        # Common IsaacLab knobs (used by many task configs).
        resampling_time_range: tuple[float, float] = (1.0, 1.0)
        debug_vis: bool = False

    class CommandTerm:
        """Minimal base class for IsaacLab command terms."""

        cfg: Any

        def __init__(self, cfg: Any, env: Any) -> None:
            self.cfg = cfg
            self._env = env
            self.device = env.device
            self.num_envs = env.num_envs
            self.metrics: dict[str, torch.Tensor] = {}

        def reset(self, env_ids: torch.Tensor):
            return {}

        def compute(self, dt: float | None = None):
            update = getattr(self, "_update_command", None)
            if callable(update):
                update()

    managers.SceneEntityCfg = SceneEntityCfg
    managers.ObservationTermCfg = ObservationTermCfg
    managers.ObservationGroupCfg = ObservationGroupCfg
    managers.RewardTermCfg = RewardTermCfg
    managers.TerminationTermCfg = TerminationTermCfg
    managers.EventTermCfg = EventTermCfg
    managers.CommandTermCfg = CommandTermCfg
    managers.CommandTerm = CommandTerm

    sys.modules["isaaclab.managers"] = managers
    isaaclab.managers = managers

    # ------------------------------------------------------------------
    # isaaclab.sim (config-only stubs)
    # ------------------------------------------------------------------
    sim = _new_module("isaaclab.sim")

    @configclass
    class RigidBodyMaterialCfg:
        friction_combine_mode: str | None = None
        restitution_combine_mode: str | None = None
        static_friction: float | None = None
        dynamic_friction: float | None = None
        restitution: float | None = None

    @configclass
    class MdlFileCfg:
        mdl_path: str = ""
        project_uvw: bool = False

    @configclass
    class DistantLightCfg:
        color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        intensity: float = 1.0

    @configclass
    class DomeLightCfg:
        color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        intensity: float = 1.0

    @configclass
    class RigidBodyPropertiesCfg:
        disable_gravity: bool = False
        retain_accelerations: bool = False
        linear_damping: float = 0.0
        angular_damping: float = 0.0
        max_linear_velocity: float = 0.0
        max_angular_velocity: float = 0.0
        max_depenetration_velocity: float = 0.0

    @configclass
    class ArticulationRootPropertiesCfg:
        enabled_self_collisions: bool = True
        solver_position_iteration_count: int = 0
        solver_velocity_iteration_count: int = 0

    @configclass
    class UrdfFileCfg:
        fix_base: bool = False
        replace_cylinders_with_capsules: bool = False
        asset_path: str = ""
        activate_contact_sensors: bool = False
        rigid_props: RigidBodyPropertiesCfg | None = None
        articulation_props: ArticulationRootPropertiesCfg | None = None
        joint_drive: Any | None = None

    @configclass
    class UrdfConverterCfg:
        @configclass
        class JointDriveCfg:
            @configclass
            class PDGainsCfg:
                stiffness: float = 0.0
                damping: float = 0.0

            gains: PDGainsCfg = PDGainsCfg()

    sim.RigidBodyMaterialCfg = RigidBodyMaterialCfg
    sim.MdlFileCfg = MdlFileCfg
    sim.DistantLightCfg = DistantLightCfg
    sim.DomeLightCfg = DomeLightCfg
    sim.RigidBodyPropertiesCfg = RigidBodyPropertiesCfg
    sim.ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg
    sim.UrdfFileCfg = UrdfFileCfg
    sim.UrdfConverterCfg = UrdfConverterCfg

    sys.modules["isaaclab.sim"] = sim
    isaaclab.sim = sim

    # ------------------------------------------------------------------
    # isaaclab.actuators
    # ------------------------------------------------------------------
    actuators = _new_module("isaaclab.actuators")

    @configclass
    class ImplicitActuatorCfg:
        joint_names_expr: list[str] = MISSING
        effort_limit_sim: dict[str, float] | float = 0.0
        velocity_limit_sim: dict[str, float] | float = 0.0
        stiffness: dict[str, float] | float = 0.0
        damping: dict[str, float] | float = 0.0
        armature: dict[str, float] | float = 0.0

    class ImplicitActuator:  # pragma: no cover - used only by optional task-side actuator models
        cfg: Any

        def __init__(self, cfg: Any, *args, **kwargs):
            self.cfg = cfg
            device = kwargs.get("device", kwargs.get("_device", "cpu"))
            self._device = torch.device(device) if not isinstance(device, torch.device) else device
            self._num_envs = int(kwargs.get("num_envs", kwargs.get("_num_envs", 1)))

        def reset(self, env_ids=None):
            return None

        def compute(self, control_action: Any, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
            # Minimal pass-through; real IsaacLab computes PD torques here.
            return control_action

    actuators.ImplicitActuatorCfg = ImplicitActuatorCfg
    actuators.ImplicitActuator = ImplicitActuator
    sys.modules["isaaclab.actuators"] = actuators
    isaaclab.actuators = actuators

    # ------------------------------------------------------------------
    # isaaclab.assets (config-only stubs + runtime type aliases)
    # ------------------------------------------------------------------
    assets = _new_module("isaaclab.assets")

    @configclass
    class AssetBaseCfg:
        prim_path: str = ""
        spawn: Any | None = None

    # These runtime types are only used for type-hints in MDP terms.
    class Articulation:  # pragma: no cover - type shim only
        pass

    class RigidObject:  # pragma: no cover - type shim only
        pass

    assets.AssetBaseCfg = AssetBaseCfg
    assets.Articulation = Articulation
    assets.RigidObject = RigidObject
    sys.modules["isaaclab.assets"] = assets
    isaaclab.assets = assets

    # Submodule: isaaclab.assets.articulation
    assets_articulation = _new_module("isaaclab.assets.articulation")

    @configclass
    class ArticulationCfg:
        @configclass
        class InitialStateCfg:
            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            joint_pos: dict[str, float] = {}
            joint_vel: dict[str, float] = {}

        prim_path: str = ""
        spawn: Any | None = None
        init_state: InitialStateCfg = InitialStateCfg()
        soft_joint_pos_limit_factor: float = 0.9
        actuators: dict[str, Any] = {}

    assets_articulation.ArticulationCfg = ArticulationCfg
    sys.modules["isaaclab.assets.articulation"] = assets_articulation
    assets.articulation = assets_articulation

    # Allow `from isaaclab.assets import ArticulationCfg`
    assets.ArticulationCfg = ArticulationCfg

    # ------------------------------------------------------------------
    # isaaclab.sensors
    # ------------------------------------------------------------------
    sensors = _new_module("isaaclab.sensors")

    @configclass
    class ContactSensorCfg:
        prim_path: str = ""
        history_length: int = 1
        track_air_time: bool = False
        force_threshold: float = 0.0
        debug_vis: bool = False
        update_period: float | None = None

    class ContactSensor:  # pragma: no cover - optional API surface
        cfg: ContactSensorCfg

        def __init__(self, cfg: ContactSensorCfg):
            self.cfg = cfg

    sensors.ContactSensorCfg = ContactSensorCfg
    sensors.ContactSensor = ContactSensor
    sys.modules["isaaclab.sensors"] = sensors
    isaaclab.sensors = sensors

    # ------------------------------------------------------------------
    # isaaclab.terrains
    # ------------------------------------------------------------------
    terrains = _new_module("isaaclab.terrains")

    @configclass
    class TerrainImporterCfg:
        prim_path: str = ""
        terrain_type: str = ""
        collision_group: int | None = None
        physics_material: Any | None = None
        visual_material: Any | None = None

    terrains.TerrainImporterCfg = TerrainImporterCfg
    sys.modules["isaaclab.terrains"] = terrains
    isaaclab.terrains = terrains

    # ------------------------------------------------------------------
    # isaaclab.scene
    # ------------------------------------------------------------------
    scene = _new_module("isaaclab.scene")

    @configclass
    class InteractiveSceneCfg:
        num_envs: int = 1
        env_spacing: float = 1.0

    scene.InteractiveSceneCfg = InteractiveSceneCfg
    sys.modules["isaaclab.scene"] = scene
    isaaclab.scene = scene

    # ------------------------------------------------------------------
    # isaaclab.markers (debug visualization no-ops)
    # ------------------------------------------------------------------
    markers = _new_module("isaaclab.markers")

    @configclass
    class _MarkerCfg:
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @configclass
    class VisualizationMarkersCfg:
        prim_path: str = ""
        markers: dict[str, Any] = {"frame": _MarkerCfg()}

    class VisualizationMarkers:  # pragma: no cover - no-op on non-isaac backends
        def __init__(self, cfg: VisualizationMarkersCfg):
            self.cfg = cfg

        def set_visibility(self, visible: bool):
            return None

        def visualize(self, *args, **kwargs):
            return None

    markers.VisualizationMarkers = VisualizationMarkers
    markers.VisualizationMarkersCfg = VisualizationMarkersCfg
    sys.modules["isaaclab.markers"] = markers
    isaaclab.markers = markers

    markers_config = _new_module("isaaclab.markers.config")
    markers_config.FRAME_MARKER_CFG = VisualizationMarkersCfg()
    sys.modules["isaaclab.markers.config"] = markers_config
    markers.config = markers_config

    # ------------------------------------------------------------------
    # isaaclab.envs + isaaclab.envs.mdp
    # ------------------------------------------------------------------
    envs = _new_module("isaaclab.envs")

    @configclass
    class _PhysxCfg:
        gpu_max_rigid_patch_count: int = 0

    @configclass
    class _SimCfg:
        dt: float = 0.005
        render_interval: int = 1
        device: str = "cpu"
        physics_material: Any | None = None
        physx: _PhysxCfg = _PhysxCfg()

    @configclass
    class _ViewerCfg:
        eye: tuple[float, float, float] = (0.0, 0.0, 0.0)
        origin_type: str | None = None
        asset_name: str | None = None

    @configclass
    class ManagerBasedRLEnvCfg:
        # Common knobs
        seed: int | None = None
        decimation: int = 1
        episode_length_s: float = 1.0
        action_clip: float | None = None

        # Sub-configs
        sim: _SimCfg = _SimCfg()
        viewer: _ViewerCfg = _ViewerCfg()

    envs.ManagerBasedRLEnvCfg = ManagerBasedRLEnvCfg
    sys.modules["isaaclab.envs"] = envs
    isaaclab.envs = envs

    envs_mdp = _new_module("isaaclab.envs.mdp")

    @configclass
    class JointPositionActionCfg:
        asset_name: str = "robot"
        joint_names: list[str] = [".*"]
        use_default_offset: bool = True
        scale: Any = 1.0
        clip: tuple[float, float] | None = None

    @configclass
    class JointVelocityActionCfg:
        asset_name: str = "robot"
        joint_names: list[str] = [".*"]
        scale: Any = 1.0
        clip: tuple[float, float] | None = None

    @configclass
    class JointEffortActionCfg:
        asset_name: str = "robot"
        joint_names: list[str] = [".*"]
        scale: Any = 1.0
        clip: tuple[float, float] | None = None

    def generated_commands(env: Any, command_name: str) -> torch.Tensor:
        return env.command_manager.get_command(command_name)

    def base_lin_vel(env: Any) -> torch.Tensor:
        art = env.scene["robot"]
        quat = art.data.root_quat_w
        vel = art.data.root_lin_vel_w
        return quat_apply_inverse(quat, vel)

    def base_ang_vel(env: Any) -> torch.Tensor:
        art = env.scene["robot"]
        quat = art.data.root_quat_w
        vel = art.data.root_ang_vel_w
        return quat_apply_inverse(quat, vel)

    def joint_pos_rel(env: Any) -> torch.Tensor:
        art = env.scene["robot"]
        return art.data.joint_pos - art.data.default_joint_pos

    def joint_vel_rel(env: Any) -> torch.Tensor:
        art = env.scene["robot"]
        return art.data.joint_vel - art.data.default_joint_vel

    def last_action(env: Any) -> torch.Tensor:
        action = getattr(env.action_manager, "action", None)
        if action is None:
            return torch.zeros((env.num_envs, 0), device=env.device)
        return action

    def action_rate_l2(env: Any) -> torch.Tensor:
        action = getattr(env.action_manager, "action", None)
        prev = getattr(env.action_manager, "prev_action", None)
        if action is None or prev is None:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.sum(torch.square(action - prev), dim=1)

    def time_out(env: Any) -> torch.Tensor:
        return env.episode_length_buf >= env.max_episode_length

    def joint_pos_limits(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos
        limits = asset.data.soft_joint_pos_limits
        out_of_limits = -(joint_pos - limits[:, :, 0]).clamp(max=0.0)
        out_of_limits += (joint_pos - limits[:, :, 1]).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def undesired_contacts(env: Any, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        if sensor.data is None:
            raise RuntimeError(f"Contact sensor '{sensor_cfg.name}' has no data.")
        body_ids, _ = sensor.find_bodies(sensor_cfg.body_names or ".*", preserve_order=True)
        ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
        forces = sensor.data.net_forces_w_history[:, :, ids, :].norm(dim=-1).max(dim=1)[0]
        is_contact = forces > float(threshold)
        return torch.sum(is_contact, dim=1)

    def push_by_setting_velocity(env: Any, env_ids: torch.Tensor, velocity_range: dict[str, tuple[float, float]]):
        art = env.scene["robot"]
        root = art.data.root_state_w.clone()
        device = root.device
        ids = env_ids.to(device=device, dtype=torch.long)
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys], device=device, dtype=root.dtype)
        samples = sample_uniform(ranges[:, 0], ranges[:, 1], (ids.numel(), 6), device=str(device))
        root[ids, 7:10] = samples[:, 0:3]
        root[ids, 10:13] = samples[:, 3:6]
        art.write_root_state_to_sim(root[ids], env_ids=ids)

    def randomize_rigid_body_material(*args, **kwargs):  # pragma: no cover - no-op stub
        return None

    envs_mdp.JointPositionActionCfg = JointPositionActionCfg
    envs_mdp.JointVelocityActionCfg = JointVelocityActionCfg
    envs_mdp.JointEffortActionCfg = JointEffortActionCfg
    envs_mdp.generated_commands = generated_commands
    envs_mdp.base_lin_vel = base_lin_vel
    envs_mdp.base_ang_vel = base_ang_vel
    envs_mdp.joint_pos_rel = joint_pos_rel
    envs_mdp.joint_vel_rel = joint_vel_rel
    envs_mdp.last_action = last_action
    envs_mdp.action_rate_l2 = action_rate_l2
    envs_mdp.time_out = time_out
    envs_mdp.joint_pos_limits = joint_pos_limits
    envs_mdp.undesired_contacts = undesired_contacts
    envs_mdp.push_by_setting_velocity = push_by_setting_velocity
    envs_mdp.randomize_rigid_body_material = randomize_rigid_body_material

    sys.modules["isaaclab.envs.mdp"] = envs_mdp
    envs.mdp = envs_mdp

    # Submodule: isaaclab.envs.mdp.events (only pieces used by BeyondMimic)
    envs_mdp_events = _new_module("isaaclab.envs.mdp.events")

    def _randomize_prop_by_op(
        values: torch.Tensor,
        distribution_params: tuple[float, float],
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor | slice,
        *,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> torch.Tensor:
        low, high = float(distribution_params[0]), float(distribution_params[1])
        if distribution == "uniform":
            noise = sample_uniform(low, high, (env_ids.numel(), values.shape[1]), device=str(values.device))
        else:
            noise = sample_uniform(low, high, (env_ids.numel(), values.shape[1]), device=str(values.device))
        out = values.clone()
        if operation == "add":
            out[env_ids] = out[env_ids] + noise
        elif operation == "scale":
            out[env_ids] = out[env_ids] * noise
        else:
            out[env_ids] = noise.abs()
        return out

    envs_mdp_events._randomize_prop_by_op = _randomize_prop_by_op
    sys.modules["isaaclab.envs.mdp.events"] = envs_mdp_events
    envs_mdp.events = envs_mdp_events
