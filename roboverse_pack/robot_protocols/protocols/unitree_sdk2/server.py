from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.setup_util import get_sim_handler_class
from roboverse_pack.robot_protocols.core.server import RobotProtocolServer, ServerConfig
from roboverse_pack.robot_protocols.core.sim_adapter import MetaSimAdapter, SimTiming
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.actuation import UnitreeLowCmdActuationModel
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.codec import AutoRemoteConfig, UnitreeSdk2Codec
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.spec_registry import get_unitree_profile
from roboverse_pack.robot_protocols.protocols.unitree_sdk2.transport import UnitreeSdk2DdsTransport


@dataclass
class UnitreeServerArgs:
    """Arguments for building a Unitree SDK2 server."""

    sim: str
    robot: str
    dt: float = 0.005
    headless: bool = True
    domain_id: int = 1
    iface: str = "lo"
    realtime: bool = True
    auto_remote: bool = False


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

    scenario = ScenarioCfg(
        robots=[args.robot],
        objects=[],
        cameras=[],
        num_envs=1,
        simulator=args.sim,
        headless=args.headless,
        env_spacing=2.5,
        decimation=1,
        sim_params=SimParamCfg(dt=args.dt, substeps=1),
    )
    scenario.__post_init__()

    handler_class = get_sim_handler_class(SimType(args.sim))
    handler = handler_class(scenario, None)
    handler.launch()

    adapter = MetaSimAdapter(handler, robot_name=args.robot, timing=SimTiming(dt=args.dt, realtime=False))

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
        protocol_to_sorted=protocol_to_sorted, torque_limits_protocol=limits_protocol
    )

    server = RobotProtocolServer(
        adapter=adapter,
        transport=transport,
        codec=codec,
        actuation=actuation,
        config=ServerConfig(dt=args.dt, realtime=args.realtime),
    )
    return server
