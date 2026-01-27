from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import logging

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_pack.robot_protocols.protocols.unitree_sdk2.server import UnitreeServerArgs, build_unitree_sdk2_server

logger = logging.getLogger(__name__)


def _parse_args() -> UnitreeServerArgs:
    parser = argparse.ArgumentParser(description="Run a Unitree SDK2 DDS protocol server backed by RoboVerse sims.")
    parser.add_argument("--sim", type=str, required=True, help="Simulator backend (e.g. isaacgym, mujoco, newton).")
    parser.add_argument("--robot", type=str, default="g1_dof29", help="Robot config name (default: g1_dof29).")
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Physics timestep (seconds). Defaults to the task dt when --match-task is set, else 0.005.",
    )
    parser.add_argument("--headless", action="store_true", help="Run simulator headless.")
    parser.add_argument("--domain-id", type=int, default=1, help="DDS domain id (default: 1).")
    parser.add_argument("--iface", type=str, default="lo", help="Network interface (default: lo).")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not sleep to match real-time; run as fast as possible.",
    )
    parser.add_argument(
        "--auto-remote",
        action="store_true",
        help="Auto-press START/A in LowState.wireless_remote for deploy_real.py compatibility.",
    )
    parser.add_argument(
        "--match-task",
        type=str,
        default=None,
        help=(
            "Align sim params + initial pose to a RoboVerse task env. Accepts a registry key "
            "(e.g. 'unitree_rl.walk_g1_dof29') or 'module:TaskClass'."
        ),
    )
    parser.add_argument(
        "--no-task-initial-state",
        action="store_true",
        help="Do not apply the task's initial root/joint state when --match-task is set.",
    )
    parser.add_argument(
        "--actuation-gains",
        choices=["cmd", "robot_cfg"],
        default=None,
        help=(
            "Gain source for LowCmd impedance control. "
            "cmd=use kp/kd from LowCmd; robot_cfg=ignore LowCmd gains and use RobotCfg stiffness/damping. "
            "Defaults to robot_cfg when --match-task is set, else cmd."
        ),
    )
    parser.add_argument(
        "--no-standby",
        action="store_true",
        help="Disable server-side standby controller (default is enabled to prevent falling before the controller is ready).",
    )
    parser.add_argument(
        "--elastic-band",
        action="store_true",
        help="Enable a MuJoCo-style elastic-band safety harness that auto-releases when protocol control becomes active.",
    )
    parser.add_argument(
        "--elastic-band-release-time",
        type=float,
        default=1.0,
        help="Seconds to ramp elastic-band force to zero after protocol takeover (default: 1.0).",
    )
    parser.add_argument(
        "--standby-mode",
        type=str,
        choices=["policy"],
        default="policy",
        help="Standby controller type used before an active LowCmd arrives (only: policy).",
    )
    parser.add_argument(
        "--standby-policy-config",
        type=str,
        default=None,
        help="Deploy config YAML (e.g. scripts/unitree_deploy/configs/g1_dof29.yaml) for standby_mode=policy.",
    )
    parser.add_argument(
        "--standby-policy-path",
        type=str,
        default=None,
        help="Override TorchScript policy path for standby_mode=policy (defaults to policy_path in the YAML).",
    )
    parser.add_argument(
        "--standby-warmup-time",
        type=float,
        default=0.0,
        help="Seconds to ramp the standby policy action_scale from 0 -> full (default: 0).",
    )
    parser.add_argument(
        "--standby-cmd-timeout",
        type=float,
        default=None,
        help="If set, revert to standby if no LowCmd received for this many seconds.",
    )
    ns = parser.parse_args()

    return UnitreeServerArgs(
        sim=ns.sim,
        robot=ns.robot,
        dt=ns.dt,
        headless=ns.headless,
        domain_id=ns.domain_id,
        iface=ns.iface,
        realtime=not ns.no_realtime,
        auto_remote=ns.auto_remote,
        match_task=ns.match_task,
        apply_task_initial_state=not ns.no_task_initial_state,
        actuation_gains=ns.actuation_gains,
        elastic_band=ns.elastic_band,
        elastic_band_release_time_s=ns.elastic_band_release_time,
        standby=not ns.no_standby,
        standby_mode=ns.standby_mode,
        standby_policy_config=ns.standby_policy_config,
        standby_policy_path=ns.standby_policy_path,
        standby_warmup_time_s=ns.standby_warmup_time,
        standby_cmd_timeout_s=ns.standby_cmd_timeout,
    )


def main() -> None:
    args = _parse_args()
    server = build_unitree_sdk2_server(args)
    server.start()

    logger.info(
        "Unitree SDK2 sim server started (sim=%s robot=%s dt=%.4f domain_id=%d iface=%s realtime=%s auto_remote=%s match_task=%s actuation_gains=%s standby=%s standby_mode=%s).",
        args.sim,
        args.robot,
        float(args.dt),
        args.domain_id,
        args.iface,
        args.realtime,
        args.auto_remote,
        args.match_task,
        args.actuation_gains,
        args.standby,
        args.standby_mode,
    )

    try:
        server.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
