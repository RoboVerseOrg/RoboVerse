from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.setup_util import get_sim_handler_class
from roboverse_pack.protocol_sim.core.elastic_band import ElasticBandAssist, ElasticBandConfig
from roboverse_pack.protocol_sim.core.server import RobotProtocolServer, ServerConfig, StandbyConfig
from roboverse_pack.protocol_sim.core.sim_adapter import MetaSimAdapter, SimTiming
from roboverse_pack.protocol_sim.core.task_alignment import apply_task_initial_state, load_task_alignment_spec
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.actuation import UnitreeLowCmdActuationModel
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.codec import AutoRemoteConfig, UnitreeSdk2Codec
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.spec_registry import get_unitree_profile
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.standby_policy import UnitreeStandbyPolicyController
from roboverse_pack.protocol_sim.protocols.unitree_sdk2.transport import UnitreeSdk2DdsTransport


@dataclass
class UnitreeServerArgs:
    """Arguments for building a Unitree SDK2 server."""

    sim: str
    robot: str
    dt: float | None = None
    headless: bool = True
    domain_id: int = 1
    iface: str = "lo"
    realtime: bool = True
    auto_remote: bool = False
    # Task-alignment (recommended for debugging mismatches vs RoboVerse task envs).
    match_task: str | None = None
    apply_task_initial_state: bool = True
    # Actuation gains source.
    # - None: defaults to "robot_cfg" when match_task is set, else "cmd"
    # - "cmd": use gains carried in LowCmd
    # - "robot_cfg": ignore LowCmd gains and use RobotCfg actuator stiffness/damping
    actuation_gains: str | None = None
    elastic_band: bool = False
    elastic_band_height_m: float = 2.0
    elastic_band_length_m: float = 0.0
    elastic_band_key_step_m: float = 0.1
    elastic_band_release_time_s: float = 1.0
    standby: bool = True
    standby_mode: str = "policy"  # only "policy" is supported for Unitree SDK2
    standby_policy_config: str | None = None
    standby_policy_path: str | None = None
    standby_warmup_time_s: float = 0.0
    standby_cmd_timeout_s: float | None = None


def build_unitree_sdk2_server(args: UnitreeServerArgs) -> RobotProtocolServer:
    """Build and configure a RobotProtocolServer for Unitree SDK2."""
    profile = get_unitree_profile(args.robot)

    try:
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

        if profile.msg_type == "hg":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
        else:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowStateDefault
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "unitree_sdk2py is required to build a Unitree SDK2 protocol server. "
            "Install unitree_sdk2_python or activate the appropriate environment."
        ) from exc

    task_spec = None
    if args.match_task is not None:
        task_spec = load_task_alignment_spec(args.match_task)

    # Default dt: task dt if aligning, else 0.005 (unitree_mujoco default).
    sim_dt = float(
        args.dt if args.dt is not None else (task_spec.scenario.sim_params.dt if task_spec is not None else 0.005)
    )
    # Mutate args for downstream consumers (CLI logging).
    args.dt = sim_dt

    if task_spec is not None:
        # Start from the task's ScenarioCfg and override the runtime knobs needed for a protocol server.
        scenario = copy.deepcopy(task_spec.scenario)
        scenario.update(
            simulator=args.sim,
            headless=args.headless,
            num_envs=1,
            decimation=1,
        )
        scenario.sim_params.dt = sim_dt
        task_robot_names = [getattr(r, "name", None) for r in scenario.robots]
        if args.robot not in task_robot_names:
            raise ValueError(
                f"--match-task '{args.match_task}' provides robots={task_robot_names}, but --robot is '{args.robot}'."
            )
        if task_robot_names and task_robot_names[0] != args.robot:
            # Prefer the requested robot as index 0 to match downstream assumptions.
            scenario.robots = [next(r for r in scenario.robots if getattr(r, "name", None) == args.robot)]
    else:
        scenario = ScenarioCfg(
            robots=[args.robot],
            objects=[],
            cameras=[],
            num_envs=1,
            simulator=args.sim,
            headless=args.headless,
            env_spacing=2.5,
            decimation=1,
            sim_params=SimParamCfg(dt=sim_dt, substeps=1),
        )
        scenario.__post_init__()

    handler_class = get_sim_handler_class(SimType(args.sim))
    handler = handler_class(scenario, None)
    handler.launch()

    if task_spec is not None and args.apply_task_initial_state and task_spec.env_cfg is not None:
        apply_task_initial_state(
            handler=handler,
            robot_name=args.robot,
            env_cfg=task_spec.env_cfg,
            pos_fallback=tuple(scenario.robots[0].default_position),
        )

    adapter = MetaSimAdapter(handler, robot_name=args.robot, timing=SimTiming(dt=sim_dt, realtime=False))

    # Map protocol motor order -> simulator sorted joint order.
    sorted_names = adapter.joint_names_sorted
    protocol_to_sorted = []
    for jn in profile.motor_names:
        if jn not in sorted_names:
            raise ValueError(f"Unitree profile expects joint '{jn}', but handler joint set is {sorted_names}.")
        protocol_to_sorted.append(sorted_names.index(jn))

    # Torque limits from RobotCfg actuators (fallback to inf).
    robot_cfg = scenario.robots[0]
    limits_sorted = []
    for jn in sorted_names:
        actuator = robot_cfg.actuators.get(jn)
        lim = None if actuator is None else getattr(actuator, "effort_limit_sim", None)
        limits_sorted.append(float(lim) if lim is not None else float("inf"))
    limits_sorted = np.asarray(limits_sorted, dtype=np.float32)
    limits_protocol = limits_sorted[protocol_to_sorted]

    # PD gains from RobotCfg in simulator sorted order.
    kp_sorted = []
    kd_sorted = []
    for jn in sorted_names:
        actuator = robot_cfg.actuators.get(jn)
        kp_sorted.append(float(getattr(actuator, "stiffness", 0.0) or 0.0) if actuator is not None else 0.0)
        kd_sorted.append(float(getattr(actuator, "damping", 0.0) or 0.0) if actuator is not None else 0.0)
    kp_sorted = np.asarray(kp_sorted, dtype=np.float32)
    kd_sorted = np.asarray(kd_sorted, dtype=np.float32)

    gains_mode = args.actuation_gains
    if gains_mode is None:
        gains_mode = "robot_cfg" if task_spec is not None else "cmd"
    if gains_mode not in ("cmd", "robot_cfg"):
        raise ValueError(f"Unknown actuation_gains '{args.actuation_gains}'. Expected 'cmd' or 'robot_cfg'.")

    kp_override_protocol = None
    kd_override_protocol = None
    if gains_mode == "robot_cfg":
        kp_override_protocol = kp_sorted[protocol_to_sorted]
        kd_override_protocol = kd_sorted[protocol_to_sorted]

    transport = UnitreeSdk2DdsTransport(domain_id=args.domain_id, iface=args.iface, profile=profile)

    codec = UnitreeSdk2Codec(
        profile=profile,
        protocol_to_sorted=protocol_to_sorted,
        lowstate_factory=LowStateDefault,
        sportstate_factory=unitree_go_msg_dds__SportModeState_,
        auto_remote=AutoRemoteConfig(enabled=args.auto_remote),
        gravity_world=scenario.gravity,
    )
    actuation = UnitreeLowCmdActuationModel(
        protocol_to_sorted=protocol_to_sorted,
        torque_limits_protocol=limits_protocol,
        kp_override_protocol=kp_override_protocol,
        kd_override_protocol=kd_override_protocol,
    )

    standby_controller = None
    standby_cfg = StandbyConfig(enabled=False)
    if args.standby:
        standby_cfg = StandbyConfig(
            enabled=True,
            cmd_timeout_s=args.standby_cmd_timeout_s,
            revert_to_standby_on_timeout=args.standby_cmd_timeout_s is not None,
            required_active_cmds=100,
        )

        if args.standby_mode != "policy":
            raise ValueError(
                f"Unsupported --standby-mode '{args.standby_mode}'. Only 'policy' is supported for Unitree SDK2."
            )

        # Use the same TorchScript policy as deploy_real.py, but apply PD gains from RobotCfg
        # to match RoboVerse task evaluation behavior.
        cfg_path = args.standby_policy_config
        if cfg_path is None:
            if args.robot == "g1_dof29":
                cfg_path = "scripts/unitree_deploy/configs/g1_dof29.yaml"
            else:
                raise ValueError(f"--standby-policy-config is required for robot='{args.robot}'.")

        standby_controller = UnitreeStandbyPolicyController(
            sim_dt=sim_dt,
            joint_names_sorted=sorted_names,
            torque_limits_sorted=limits_sorted,
            kp_sorted=kp_sorted,
            kd_sorted=kd_sorted,
            config_path=cfg_path,
            policy_path=args.standby_policy_path,
            warmup_time_s=args.standby_warmup_time_s,
        )

    assist = None
    if args.elastic_band:
        assist = ElasticBandAssist(
            handler=handler,
            robot_name=args.robot,
            cfg=ElasticBandConfig(
                point=(0.0, 0.0, float(args.elastic_band_height_m)),
                length=float(args.elastic_band_length_m),
                release_time_s=float(args.elastic_band_release_time_s),
            ),
        )

    server = RobotProtocolServer(
        adapter=adapter,
        transport=transport,
        codec=codec,
        actuation=actuation,
        config=ServerConfig(dt=sim_dt, realtime=args.realtime),
        standby_controller=standby_controller,
        standby_config=standby_cfg,
        assist=assist,
    )
    return server
