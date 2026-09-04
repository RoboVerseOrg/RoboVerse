"""Configuration classes for simulator parameters."""

from __future__ import annotations

from typing import Literal

from metasim.utils import configclass


@configclass
class SimParamCfg:
    """Simulation parameters cfg.

    This class defines the parameters for the simulator.
    It is important to ensure that each task is configured with appropriate simulation
    parameters to avoid divergence or unexpected results.

    Reference for IsaacGym: https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
    """

    ## Simulation
    dt: float | None = None
    """The time-step of the simulation. If None, the default value in the original simulator will be used. The default value for each simulator is:
    - IsaacGym: 1/60
    - IsaacLab: 1/60
    - MuJoCo: 1/500
    - PyBullet: 1/240
    """

    ## Physics
    bounce_threshold_velocity: float = 0.2
    contact_offset: float = 0.001
    num_position_iterations: int = 8
    num_velocity_iterations: int = 1
    friction_correlation_distance: float = 0.0005
    friction_offset_threshold: float = 0.001
    replace_cylinder_with_capsule: bool = False
    rest_offset: float = 0.0
    solver_type: int = 1
    substeps: int = 1  # for IsaacGym
    max_depenetration_velocity: float = 1.0
    default_buffer_size_multiplier: int = 2

    ## SAPIEN-specific PhysX scene parameters.
    # All default to None = "leave the SAPIEN SceneConfig default untouched", so
    # existing tasks are unaffected. Set them to match an upstream env — e.g.
    # SimplerEnv / ManiSkill2-real2sim uses solver_iterations=25, enable_tgs=True,
    # contact_offset=0.02, enable_pcm=False, solver_velocity_iterations=1, friction 1.0.
    sapien_enable_tgs: bool | None = None
    sapien_enable_pcm: bool | None = None
    sapien_default_static_friction: float | None = None
    sapien_default_dynamic_friction: float | None = None
    sapien_default_restitution: float | None = None
    # When True, the sapien handler additionally applies contact_offset and the
    # solver iteration counts from this cfg onto the SAPIEN SceneConfig. When None/False
    # it keeps its historical behaviour (only gravity + timestep set).
    sapien_apply_scene_solver: bool | None = None
    # When True, the sapien handler skips its built-in default lights so a task can install its
    # own (e.g. SimplerEnv visual-matching lighting) via the exposed scene after launch.
    sapien_disable_default_lights: bool | None = None
    # When True, the sapien URDF loader loads multiple convex collision meshes per link
    # (loader.load_multiple_collisions_from_file). Needed for assets whose collision geometry is
    # split into several convex pieces (e.g. the SimplerEnv google_robot). Default None = legacy.
    sapien_load_multiple_collisions: bool | None = None

    # --- ManiSkill-faithful PhysX recipe (all None/False = legacy, existing tasks unaffected) ---
    # When True, the sapien3 handler applies the FULL PhysX config globally (physx.set_shape_config /
    # set_body_config / set_scene_config / set_default_material) before creating the scene, mirroring
    # ManiSkill's BaseEnv._set_scene_config. This is the correct SAPIEN-3 path (the legacy
    # SceneConfig.solver_iterations attribute no longer exists); it reads contact_offset, rest_offset,
    # num_position_iterations, num_velocity_iterations from this cfg plus the fields below.
    sapien_apply_global_physx: bool | None = None
    sapien_sleep_threshold: float | None = None
    sapien_bounce_threshold: float | None = None
    sapien_enable_ccd: bool | None = None
    sapien_enable_enhanced_determinism: bool | None = None
    sapien_enable_friction_every_iteration: bool | None = None
    # When True, the handler disables gravity on every robot link (link.disable_gravity=True) and
    # skips its built-in gravity-compensation passive force. This is how ManiSkill holds the arm and
    # is required for dynamics parity with it. Default None/False keeps the passive-force behaviour.
    sapien_disable_robot_gravity: bool | None = None
    # Altitude of the auto-added ground plane. None keeps the historical 0.0. ManiSkill puts the
    # ground far below a kinematic table box, e.g. -0.9196429.
    sapien_ground_altitude: float | None = None
    # When set (e.g. "force"), robot joint drives are created with this drive mode and the actuator's
    # effort_limit_sim as the force limit. None keeps the legacy set_drive_property(stiffness, damping).
    sapien_drive_force_mode: str | None = None

    ## SuperDex-specific parameters
    # "pd": every substep applies tau = clip(kp*(q_target - q) - kd*(qd - qd_target), +-effort_limit) as an
    # external DoF force -- the same control law and clamp the MuJoCo backend's actuators implement, so
    # closed-loop behaviour matches it. "implicit": SuperDex's native constraint-based pose controller
    # (stiffer, ignores effort limits, converges in a few ms).
    superdex_control_mode: Literal["pd", "implicit"] = "pd"
    # SuperDex only: the step the implicit solver takes. None -> 5 ms (``metasim.sim.superdex.DEFAULT_SOLVER_DT``);
    # the env step (``dt * decimation``) is covered by ``round(env_step / solver_dt)`` steps of equal size, so
    # ``dt`` and ``decimation`` keep their meaning. 0.001 reproduces the pre-1.0 1 ms stepping.
    superdex_solver_dt: float | None = None

    ## MJX, Newton specific parameters
    nconmax: int | None = 512
    njmax: int | None = None
    # If None, Newton defaults to MuJoCo contacts when using SolverMuJoCo
    newton_use_mujoco_contacts: bool | None = None
    # Newton SolverMuJoCo solver iterations. None = solver default. mjlab native
    # passes (iterations=10, ls_iterations=20) for humanoid tasks.
    newton_mujoco_iterations: int | None = None
    newton_mujoco_ls_iterations: int | None = None
    # Disable contact computation in Newton's SolverMuJoCo — mjlab uses this on
    # contact-free tasks (cartpole) where rail/cart contacts would add noise.
    newton_mujoco_disable_contacts: bool = False

    ## Resource management
    num_threads: int = 0
    # XXX: these parameters should be replaced by "device" in the future
    use_gpu_pipeline: bool = True
    use_gpu: bool = True

    # ---- MuJoCo / MJX ----
    # MuJoCo / MJX: render ``scenario.lights`` (translated to MJCF lights, replacing the headlight and any
    # lights embedded in robot / object models). Off by default: ``ScenarioCfg.lights`` defaults to one
    # distant light and most scenarios never set it, so enabling this for everyone would re-light every
    # MuJoCo scene. See ``metasim/sim/mujoco/lights.py`` and docs ``features/lighting``.
    mujoco_use_scenario_lights: bool = False

    def __post_init__(self) -> None:
        """``dt`` is None (the backend's default step) or a finite step > 0.

        ``""``, ``nan``, ``inf`` and non-positive values are rejected here instead of reaching the
        physics engine, where they surface as a hang, a NaN state or a backend-specific error.
        """
        from ._validate import positive_finite_or_none, positive_int

        positive_finite_or_none("SimParamCfg", "dt", self.dt)
        positive_int("SimParamCfg", "substeps", self.substeps)
