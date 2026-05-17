"""Driver: roll out every mjlab task in raw mujoco + the MetaSim loader path.

Writes per-task artifacts to `reports/mjlab_integration/runs/<task>/` and a
top-level summary JSON suitable for the HTML report generator.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from . import inventory as inv
from .diff import diff_rollouts
from .plot import plot_ctrl, plot_trajectory_pair
from .raw_rollout import RolloutResult, raw_mujoco_rollout, sinusoid_ctrl, zero_ctrl
from .render import configure_offscreen_gl, replay_to_video


def _ctrl_for_asset(asset_name: str, nu: int, dt: float):
    """Pick a deterministic ctrl sequence that exercises the model.

    - cartpole: small sinusoid on the single motor.
    - everything else: zero ctrl (passive drop / hold), which is the most
      robust comparison for legged & arm models without policy logic.
    """
    if asset_name == "cartpole":
        return sinusoid_ctrl(nu=nu, amplitude=0.3, freq_hz=0.5, dt=dt)
    return zero_ctrl(nu)


def _save_rollout(result: RolloutResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=result.qpos,
        qvel=result.qvel,
        ctrl=result.ctrl,
        time=result.time,
        dt=np.float64(result.dt),
        decimation=np.int32(result.decimation),
        backend=np.array(result.backend),
        task_id=np.array(result.task_id),
        joint_names=np.array(result.joint_names),
        actuator_names=np.array(result.actuator_names),
    )


def run_one_task(task: inv.MjlabTask, runs_dir: Path, *, n_steps: int, render: bool) -> dict:
    """Run one task, write artifacts, return a summary dict."""
    task_dir = runs_dir / task.metasim_name
    task_dir.mkdir(parents=True, exist_ok=True)

    xml_path = task.asset.xml_path
    if not xml_path.exists():
        return {
            "task": task.metasim_name,
            "task_id": task.task_id,
            "asset": task.asset.name,
            "status": "missing_asset",
            "xml": str(xml_path),
            "error": "MJCF not found",
        }

    # Probe model to learn nu/dt
    import mujoco

    probe = mujoco.MjModel.from_xml_path(str(xml_path))
    nu = int(probe.nu)
    base_dt = float(probe.opt.timestep)
    ctrl_fn = _ctrl_for_asset(task.asset.name, nu=nu, dt=base_dt)

    summary: dict = {
        "task": task.metasim_name,
        "task_id": task.task_id,
        "asset": task.asset.name,
        "asset_xml": str(xml_path),
        "decimation": task.decimation,
        "nu": nu,
        "nq": int(probe.nq),
        "nv": int(probe.nv),
        "dt": base_dt,
        "n_steps": n_steps,
        "status": "ok",
        "artifacts": {},
    }

    try:
        raw = raw_mujoco_rollout(
            xml_path,
            n_steps=n_steps,
            decimation=task.decimation,
            ctrl_fn=ctrl_fn,
            init_joint_pos=task.init_joint_pos,
            task_id=task.metasim_name,
        )
    except Exception as e:
        summary.update(status="raw_rollout_failed", error=repr(e), traceback=traceback.format_exc())
        return summary

    _save_rollout(raw, task_dir / "raw_mujoco.npz")
    summary["artifacts"]["raw_npz"] = str(task_dir / "raw_mujoco.npz")
    summary["raw_summary"] = raw.summary()

    # MetaSim loader pathway (dm_control roundtrip).
    try:
        from .metasim_rollout import metasim_loader_rollout  # local: heavy import

        meta_loader = metasim_loader_rollout(
            xml_path,
            n_steps=n_steps,
            decimation=task.decimation,
            ctrl_fn=ctrl_fn,
            init_joint_pos=task.init_joint_pos,
            task_id=task.metasim_name,
        )
    except Exception as e:
        summary.update(status="metasim_loader_failed", error=repr(e), traceback=traceback.format_exc())
        return summary

    _save_rollout(meta_loader, task_dir / "metasim_loader.npz")
    summary["artifacts"]["metasim_loader_npz"] = str(task_dir / "metasim_loader.npz")
    summary["metasim_loader_summary"] = meta_loader.summary()

    # Diff
    try:
        diff = diff_rollouts(raw, meta_loader)
        summary["diff_raw_vs_loader"] = diff.as_row()
    except Exception as e:
        summary.update(status="diff_failed", error=repr(e), traceback=traceback.format_exc())
        return summary

    # Plots
    try:
        plot_trajectory_pair(
            raw,
            meta_loader,
            task_dir / "qpos_compare.png",
            title=f"{task.metasim_name}  —  raw mujoco vs MetaSim loader",
        )
        plot_ctrl(raw, task_dir / "ctrl.png", title=f"{task.metasim_name}  —  applied ctrl")
        summary["artifacts"]["qpos_plot"] = "qpos_compare.png"
        summary["artifacts"]["ctrl_plot"] = "ctrl.png"
    except Exception as e:
        summary["plot_error"] = repr(e)

    # Video (optional, may fail on headless hosts without GL)
    if render:
        try:
            configure_offscreen_gl()
            v = replay_to_video(xml_path, raw, task_dir / "raw_replay.mp4")
            if v:
                summary["artifacts"]["raw_video"] = Path(v).name
            v2 = replay_to_video(xml_path, meta_loader, task_dir / "metasim_loader_replay.mp4")
            if v2:
                summary["artifacts"]["metasim_loader_video"] = Path(v2).name
        except Exception as e:
            summary["render_error"] = repr(e)

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/ghr/projects/RoboVerse/reports/mjlab_integration")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--tasks", nargs="*", help="Subset of tasks (metasim names) to run.")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--only-cartpole", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    tasks = inv.ALL_TASKS
    if args.only_cartpole:
        tasks = [t for t in tasks if t.asset.name == "cartpole"]
    if args.tasks:
        tasks = [t for t in tasks if t.metasim_name in args.tasks]

    all_summaries = []
    for t in tasks:
        print(f"[mjlab_integration] >> {t.metasim_name}  ({t.task_id})  asset={t.asset.name}")
        try:
            s = run_one_task(t, runs_dir, n_steps=args.n_steps, render=not args.no_render)
        except Exception as e:
            s = {
                "task": t.metasim_name,
                "task_id": t.task_id,
                "asset": t.asset.name,
                "status": "runner_failed",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
        all_summaries.append(s)
        # Print one-line status
        diff = s.get("diff_raw_vs_loader", {})
        print(
            f"   status={s.get('status')}  "
            f"qpos_max|Δ|={diff.get('qpos_max_abs', float('nan')):.3e}  "
            f"qvel_max|Δ|={diff.get('qvel_max_abs', float('nan')):.3e}"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"[mjlab_integration] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
