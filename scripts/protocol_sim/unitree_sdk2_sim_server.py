from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import logging
import sys
import threading

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from roboverse_pack.protocol_sim.protocols.unitree_sdk2.server import UnitreeServerArgs, build_unitree_sdk2_server

logger = logging.getLogger(__name__)


def _start_elastic_band_live_tuning(server, *, key_step_m: float) -> threading.Thread | None:
    """Start a background stdin loop for live elastic-band tuning commands."""
    assist = getattr(server, "assist", None)
    if assist is None:
        return None
    if not all(
        hasattr(assist, name) for name in ("set_length", "set_anchor_height", "get_length", "get_anchor_height")
    ):
        logger.warning("Elastic-band assist does not expose live-tuning hooks; stdin tuning disabled.")
        return None
    if not hasattr(assist, "start_release"):
        logger.warning("Elastic-band assist does not support manual release; stdin tuning disabled.")
        return None
    if not sys.stdin.isatty():
        logger.info("Elastic-band live tuning disabled (stdin is not a TTY).")
        return None

    # Prefer direct key-control mode (no Enter required) to match "button-like" control.
    try:
        import termios
        import tty
    except Exception:
        termios = None
        tty = None

    if termios is not None and tty is not None:
        logger.info(
            "Elastic-band key control enabled: '7' stronger assist, '8' weaker assist, ']' anchor up, '[' anchor down, 'r' release, 's' show, 'h' help."
        )

        def _repl_keys() -> None:
            fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(fd)
            try:
                # cbreak mode keeps Ctrl+C working while allowing single-key reads.
                tty.setcbreak(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch == "":
                        return
                    if ch in ("h", "H", "?"):
                        logger.info(
                            "Elastic-band keys: 7(stronger assist, shorter rest length) 8(weaker assist, longer rest length) ](anchor up) [(anchor down) r(release band) s(show) h(help)"
                        )
                        continue
                    if ch in ("s", "S"):
                        logger.info(
                            "Elastic-band state: height=%.3f m, length=%.3f m",
                            float(assist.get_anchor_height()),
                            float(assist.get_length()),
                        )
                        continue
                    if ch == "8":
                        # Longer rest length -> weaker spring pull.
                        prev = float(assist.get_length())
                        requested = prev + float(key_step_m)
                        new = float(assist.set_length(requested))
                        logger.info(
                            "Elastic-band weaker assist (key 8): rest_length %.3f -> %.3f m",
                            prev,
                            new,
                        )
                        continue
                    if ch == "7":
                        # Shorter rest length -> stronger spring pull.
                        prev = float(assist.get_length())
                        requested = prev - float(key_step_m)
                        new = float(assist.set_length(requested))
                        clamp_note = " (clamped at 0.0 m)" if new != requested else ""
                        logger.info(
                            "Elastic-band stronger assist (key 7): rest_length %.3f -> %.3f m%s",
                            prev,
                            new,
                            clamp_note,
                        )
                        continue
                    if ch == "]":
                        prev = float(assist.get_anchor_height())
                        new = prev + float(key_step_m)
                        assist.set_anchor_height(new)
                        logger.info("Elastic-band anchor up: height %.3f -> %.3f m", prev, new)
                        continue
                    if ch == "[":
                        prev = float(assist.get_anchor_height())
                        new = prev - float(key_step_m)
                        assist.set_anchor_height(new)
                        logger.info("Elastic-band anchor down: height %.3f -> %.3f m", prev, new)
                        continue
                    if ch in ("r", "R"):
                        assist.start_release()
                        logger.info("Elastic-band manual release started (key r).")
                        continue
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

        thread = threading.Thread(target=_repl_keys, name="elastic-band-live-tuning-keys", daemon=True)
        thread.start()
        return thread

    logger.info("Elastic-band live tuning in line mode: 'length <m>', 'height <m>', 'release', 'show', 'help'.")

    def _repl_lines() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            text = line.strip()
            if not text:
                continue

            parts = text.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ("help", "h", "?"):
                logger.info("Elastic-band commands: length <m> | height <m> | release | show | help")
                continue
            if cmd == "show":
                logger.info(
                    "Elastic-band state: height=%.3f m, length=%.3f m",
                    float(assist.get_anchor_height()),
                    float(assist.get_length()),
                )
                continue
            if cmd in ("length", "len", "l"):
                if arg is None:
                    logger.warning("Usage: length <meters> (clamped to >= 0)")
                    continue
                try:
                    value = float(arg)
                except ValueError:
                    logger.warning("Invalid length value: %s", arg)
                    continue
                applied = float(assist.set_length(value))
                clamp_note = " (clamped at 0.0 m)" if applied != value else ""
                logger.info("Elastic-band rest_length set to %.3f m%s", applied, clamp_note)
                continue
            if cmd in ("height", "z"):
                if arg is None:
                    logger.warning("Usage: height <meters>")
                    continue
                try:
                    value = float(arg)
                except ValueError:
                    logger.warning("Invalid height value: %s", arg)
                    continue
                assist.set_anchor_height(value)
                logger.info("Elastic-band height set to %.3f m", value)
                continue
            if cmd in ("release", "rel", "r"):
                assist.start_release()
                logger.info("Elastic-band manual release started.")
                continue

            logger.warning("Unknown command '%s'. Use: length <m> | height <m> | release | show | help", text)

    thread = threading.Thread(target=_repl_lines, name="elastic-band-live-tuning-lines", daemon=True)
    thread.start()
    return thread


def _parse_args() -> UnitreeServerArgs:
    parser = argparse.ArgumentParser(description="Run a Unitree SDK2 DDS protocol server backed by RoboVerse sims.")
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Simulator backend (e.g. isaacgym, isaacsim, mujoco, newton).",
    )
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
        help=(
            "Enable an elastic-band safety harness (MuJoCo/IsaacGym/IsaacSim/Newton) "
            "that can be manually released with keyboard command 'r'."
        ),
    )
    parser.add_argument(
        "--elastic-band-height",
        type=float,
        default=2.0,
        help="Elastic-band anchor height in world frame (meters, default: 2.0).",
    )
    parser.add_argument(
        "--elastic-band-length",
        type=float,
        default=0.0,
        help=(
            "Elastic-band rest length (meters, default: 0.0, clamped to >= 0). Larger values reduce lift force. "
            "When running interactively, you can tune it live with keys 8/7."
        ),
    )
    parser.add_argument(
        "--elastic-band-key-step",
        type=float,
        default=0.1,
        help="Live key-control increment in meters for elastic band length/height (default: 0.1).",
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
        elastic_band_height_m=ns.elastic_band_height,
        elastic_band_length_m=ns.elastic_band_length,
        elastic_band_key_step_m=ns.elastic_band_key_step,
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

    if args.elastic_band:
        _start_elastic_band_live_tuning(server, key_step_m=float(args.elastic_band_key_step_m))

    elastic_band_height_eff = float(args.elastic_band_height_m)
    elastic_band_length_eff = float(args.elastic_band_length_m)
    assist = getattr(server, "assist", None)
    if assist is not None:
        if hasattr(assist, "get_anchor_height"):
            elastic_band_height_eff = float(assist.get_anchor_height())
        if hasattr(assist, "get_length"):
            elastic_band_length_eff = float(assist.get_length())

    logger.info(
        "Unitree SDK2 sim server started (sim=%s robot=%s dt=%.4f domain_id=%d iface=%s realtime=%s auto_remote=%s match_task=%s actuation_gains=%s standby=%s standby_mode=%s elastic_band=%s elastic_band_height=%.3f elastic_band_length=%.3f elastic_band_key_step=%.3f).",
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
        args.elastic_band,
        elastic_band_height_eff,
        elastic_band_length_eff,
        float(args.elastic_band_key_step_m),
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
