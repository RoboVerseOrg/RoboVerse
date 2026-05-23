"""Render side-by-side mp4s for every task with saved rollouts.

The runners persist raw_mujoco.npz and the metasim variant under
runs/<task>/. We replay both onto a fresh MjModel and concat into
runs/<task>/side_by_side_full.mp4 (and .loader.mp4 for the loader
sweep).

Run after `tools.mjlab_integration.full_runner` and/or
`tools.mjlab_integration.runner` so the rollouts exist on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import inventory as inv
from .raw_rollout import RolloutResult
from .render import side_by_side_video

_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")


def _load_npz(path: Path) -> RolloutResult | None:
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)

    def _list(key):
        v = npz.get(key)
        if v is None:
            return None
        return list(v.tolist())

    return RolloutResult(
        backend=str(npz["backend"].item() if "backend" in npz.files else "unknown"),
        task_id=str(npz["task_id"].item() if "task_id" in npz.files else path.stem),
        dt=float(npz["dt"].item() if "dt" in npz.files else 0.01),
        decimation=int(npz["decimation"].item() if "decimation" in npz.files else 1),
        qpos=npz["qpos"],
        qvel=npz["qvel"],
        ctrl=npz["ctrl"] if "ctrl" in npz.files else np.zeros((0, 0)),
        time=npz["time"] if "time" in npz.files else np.arange(npz["qpos"].shape[0]),
        joint_names=list(npz["joint_names"].tolist()) if "joint_names" in npz.files else [],
        actuator_names=list(npz["actuator_names"].tolist()) if "actuator_names" in npz.files else [],
        jnt_qposadr=_list("jnt_qposadr"),
        jnt_dofadr=_list("jnt_dofadr"),
        jnt_type=_list("jnt_type"),
    )


def _save_rollout(path: Path, r: RolloutResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        backend=r.backend,
        task_id=r.task_id,
        dt=r.dt,
        decimation=r.decimation,
        qpos=r.qpos,
        qvel=r.qvel,
        ctrl=r.ctrl,
        time=r.time,
        joint_names=np.array(r.joint_names, dtype=object),
        actuator_names=np.array(r.actuator_names, dtype=object),
        jnt_qposadr=np.array(r.jnt_qposadr or [], dtype=np.int64),
        jnt_dofadr=np.array(r.jnt_dofadr or [], dtype=np.int64),
        jnt_type=np.array(r.jnt_type or [], dtype=np.int64),
    )


def render_all(reports_dir: Path, n_steps: int = 200, frame_stride: int = 2) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    out_dir = reports_dir

    summary: dict = {"renders": []}
    for task in inv.ALL_TASKS:
        run_dir = out_dir / "runs" / task.metasim_name
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_npz = run_dir / "raw_mujoco.npz"
        meta_npz = run_dir / "metasim_full.npz"

        # If rollouts haven't been persisted yet, generate them now.
        if not raw_npz.exists() or not meta_npz.exists():
            import mujoco

            from .full_runner import SCENARIO_BUILDERS, _ctrl_for
            from .metasim_rollout import metasim_full_rollout
            from .raw_rollout import raw_mujoco_rollout

            probe = mujoco.MjModel.from_xml_path(str(task.asset.xml_path))
            nu = int(probe.nu)
            base_dt = float(probe.opt.timestep)
            ctrl_fn = _ctrl_for(task.asset.name, nu=nu, dt=base_dt)

            raw = raw_mujoco_rollout(
                task.asset.xml_path,
                n_steps=n_steps,
                decimation=task.decimation,
                ctrl_fn=ctrl_fn,
                init_joint_pos=task.init_joint_pos,
                task_id=task.metasim_name,
            )
            _save_rollout(raw_npz, raw)

            builder = SCENARIO_BUILDERS.get(task.metasim_name)
            if builder is None:
                continue
            scenario = builder(task)
            scenario.sim_params.dt = base_dt
            try:
                meta = metasim_full_rollout(
                    scenario,
                    n_steps=n_steps,
                    decimation=task.decimation,
                    ctrl_fn=ctrl_fn,
                    init_joint_pos=task.init_joint_pos,
                    task_id=task.metasim_name,
                )
                _save_rollout(meta_npz, meta)
            except Exception as e:
                print(f"[render_sweep] {task.metasim_name}: meta rollout failed — {e}")
                continue

        raw = _load_npz(raw_npz)
        meta = _load_npz(meta_npz)
        if raw is None or meta is None:
            print(f"[render_sweep] {task.metasim_name}: missing rollout npz")
            continue

        out_mp4 = run_dir / "side_by_side_full.mp4"
        try:
            written = side_by_side_video(
                task.asset.xml_path,
                raw,
                meta,
                out_mp4,
                width=360,
                height=270,
                camera=-1,
                frame_stride=frame_stride,
                labels=("raw mujoco", "MetaSim full"),
            )
            summary["renders"].append({"task": task.metasim_name, "mp4": written})
            print(f"[render_sweep] {task.metasim_name} -> {written}")
        except Exception as e:
            print(f"[render_sweep] {task.metasim_name}: render failed — {e}")
            summary["renders"].append({"task": task.metasim_name, "error": str(e)})

    summary_path = out_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPORTS, "mjlab_integration"))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--frame-stride", type=int, default=2)
    args = p.parse_args(argv)
    render_all(Path(args.out), n_steps=args.n_steps, frame_stride=args.frame_stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
