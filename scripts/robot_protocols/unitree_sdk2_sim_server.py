from __future__ import annotations

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
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep (seconds).")
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
    )


def main() -> None:
    args = _parse_args()
    server = build_unitree_sdk2_server(args)
    server.start()

    logger.info(
        "Unitree SDK2 sim server started (sim=%s robot=%s dt=%.4f domain_id=%d iface=%s realtime=%s auto_remote=%s).",
        args.sim,
        args.robot,
        args.dt,
        args.domain_id,
        args.iface,
        args.realtime,
        args.auto_remote,
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
