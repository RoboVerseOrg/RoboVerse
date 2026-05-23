"""End-to-end humanoid demo for mjlab robots, in two flavors:

- **pendant**: root welded to world at standing height. The robot can't
  tip over, so we can freely swing arms / cycle legs to show the PD +
  actuator + scene plumbing all works. This is what we embed in the
  report as "the robot is doing something".

- **freefall**: normal floating-base. With a scripted policy a humanoid
  collapses within ~2 s — a learned RL controller is what mjlab uses
  to maintain balance. We include this demo too, honestly labelled, to
  show MetaSim is running gravity + contact correctly.

For each robot we:
1. Build the scene via `build_stand_scenes.build_scene(...)` (cached XML).
2. Reset to the standing keyframe.
3. Drive the joints with a hand-crafted scripted policy.
4. Record state + render the rollout as mp4.

Outputs live at `reports/mjlab_integration/runs/<robot>_{pendant,freefall}/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from .scenes.build_stand_scenes import (
    G1_ACTUATORS,
    G1_STANDING_POSE,
    GO1_ACTUATORS,
    GO1_STANDING_POSE,
    YAM_ACTUATORS,
    YAM_STANDING_POSE,
    build_compiled_model,
)

DEFAULT_OUT = Path(os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")) / "mjlab_integration"


def _actuator_target_indices(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    out = []
    for jn in joint_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_act")
        if aid < 0:
            raise RuntimeError(f"actuator {jn}_act not found")
        out.append(aid)
    return out


# --- scripted policies ----------------------------------------------------


def _g1_pendant_action(t: float, base: np.ndarray, joints: list[str]) -> np.ndarray:
    """Pendant g1: free to wave arms + bend knees in a 'jumping jack' loop."""
    out = base.copy()
    arm_swing = 0.8 * math.sin(2.0 * math.pi * 0.7 * t)
    leg_bend = 0.4 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 0.5 * t))
    for i, j in enumerate(joints):
        if j == "left_shoulder_roll_joint":
            out[i] = base[i] + arm_swing
        elif j == "right_shoulder_roll_joint":
            out[i] = base[i] - arm_swing
        elif j == "left_elbow_joint":
            out[i] = base[i] + 0.3 + 0.3 * math.sin(2.0 * math.pi * 1.0 * t)
        elif j == "right_elbow_joint":
            out[i] = base[i] + 0.3 + 0.3 * math.sin(2.0 * math.pi * 1.0 * t + math.pi)
        elif j in ("left_knee_joint", "right_knee_joint"):
            out[i] = base[i] + leg_bend
        elif j in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
            out[i] = base[i] - 0.5 * leg_bend
        elif j in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            out[i] = base[i] - 0.5 * leg_bend
    return out


def _go1_pendant_action(t: float, base: np.ndarray, joints: list[str]) -> np.ndarray:
    """Pendant go1: 1 Hz diagonal-pair trot. Welded base, so legs free to swing."""
    out = base.copy()
    phase = 2.0 * math.pi * 1.0 * t
    for i, j in enumerate(joints):
        if j in ("FR_thigh_joint", "RL_thigh_joint"):
            out[i] = base[i] + 0.3 * math.sin(phase)
        elif j in ("FL_thigh_joint", "RR_thigh_joint"):
            out[i] = base[i] + 0.3 * math.sin(phase + math.pi)
        elif j in ("FR_calf_joint", "RL_calf_joint"):
            out[i] = base[i] - 0.4 * (0.5 + 0.5 * math.sin(phase))
        elif j in ("FL_calf_joint", "RR_calf_joint"):
            out[i] = base[i] - 0.4 * (0.5 + 0.5 * math.sin(phase + math.pi))
    return out


def _yam_action(t: float, base: np.ndarray, joints: list[str]) -> np.ndarray:
    out = base.copy()
    for i, j in enumerate(joints):
        if j == "joint1":
            out[i] = base[i] + 1.0 * math.sin(2.0 * math.pi * 0.3 * t)
        elif j == "joint2":
            out[i] = base[i] + 0.5 * math.sin(2.0 * math.pi * 0.3 * t + math.pi / 2)
        elif j == "joint3":
            out[i] = base[i] + 0.3 * math.sin(2.0 * math.pi * 0.5 * t)
    return out


def _hold_action(t: float, base: np.ndarray, joints: list[str]) -> np.ndarray:
    """For freefall demos: just hold the keyframe target. The robot's gravity-
    driven fall is the point."""
    return base.copy()


SCRIPTED = {
    ("g1", "pendant"): _g1_pendant_action,
    ("g1", "freefall"): _hold_action,
    ("go1", "pendant"): _go1_pendant_action,
    ("go1", "freefall"): _hold_action,
    ("yam", "pendant"): _yam_action,
    ("yam", "freefall"): _yam_action,  # yam is fixed-base regardless
}


def run_demo(robot: str, variant: str, out_dir: Path, n_seconds: float = 4.0, fps: int = 30) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    model = build_compiled_model(robot, pin_root=(variant == "pendant"))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    # mj_forward populates xpos / xfrc from qpos before the first step, so
    # the initial root-z reading reflects the actual pose rather than 0.
    mujoco.mj_forward(model, data)

    pose = {"g1": G1_STANDING_POSE, "go1": GO1_STANDING_POSE, "yam": YAM_STANDING_POSE}[robot]
    actuators = {"g1": G1_ACTUATORS, "go1": GO1_ACTUATORS, "yam": YAM_ACTUATORS}[robot]
    joint_names = [a[0] for a in actuators]
    base_target = np.array([pose[j] for j in joint_names], dtype=np.float64)
    act_ids = _actuator_target_indices(model, joint_names)
    action_fn = SCRIPTED[(robot, variant)]

    n_steps = int(n_seconds / model.opt.timestep)
    render_every = max(1, round((1.0 / fps) / model.opt.timestep))

    renderer = mujoco.Renderer(model, width=480, height=360)

    qpos_log = [data.qpos.copy()]
    qvel_log = [data.qvel.copy()]
    ctrl_log: list[np.ndarray] = []
    # Track the root body's world z so we can report trunk height regardless of
    # whether the root is on a free joint (variant=freefall) or welded (pendant).
    # mjcf compile order means body[1] is the robot's root in both variants.
    root_z_log = [float(data.xpos[1, 2])]
    frames: list[np.ndarray] = []

    for k in range(n_steps):
        t_sec = k * model.opt.timestep
        ctrl_target = action_fn(t_sec, base_target, joint_names)
        for i, aid in enumerate(act_ids):
            data.ctrl[aid] = float(ctrl_target[i])
        mujoco.mj_step(model, data)
        ctrl_log.append(data.ctrl.copy())
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        root_z_log.append(float(data.xpos[1, 2]))
        if k % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    qv_arr = np.stack(qvel_log)
    rz_arr = np.array(root_z_log)
    summary = {
        "robot": robot,
        "variant": variant,
        "n_seconds": n_seconds,
        "n_steps": n_steps,
        "nq": int(model.nq),
        "nu": int(model.nu),
        "root_z_initial": float(rz_arr[0]),
        "root_z_final": float(rz_arr[-1]),
        "root_z_settled": float(rz_arr[-int(0.5 * n_steps) :].mean()),
        "max_qvel_final_quarter": float(np.abs(qv_arr[-n_steps // 4 :]).max()),
    }
    # Stability: for pendant, root z stays near init; for freefall we don't
    # require it (the demo's point is gravity is on).
    if variant == "pendant":
        summary["stable"] = bool(
            abs(summary["root_z_final"] - summary["root_z_initial"]) < 0.05 and summary["max_qvel_final_quarter"] < 10.0
        )
    else:
        summary["stable"] = bool(summary["max_qvel_final_quarter"] < 1.0)  # = "came to rest"

    run_dir = out_dir / "runs" / f"{robot}_{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "rollout.npz",
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log) if ctrl_log else np.zeros((0, model.nu)),
        root_z=rz_arr,
    )
    try:
        import imageio

        out_mp4 = run_dir / "demo.mp4"
        imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", quality=7)
        summary["mp4"] = str(out_mp4)
    except Exception as e:
        print(f"[stand_demo] imageio failed: {e}")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--robots", nargs="*", default=["g1", "go1", "yam"])
    p.add_argument("--variants", nargs="*", default=["pendant", "freefall"])
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for r in args.robots:
        for v in args.variants:
            if r == "yam" and v == "freefall":
                continue  # yam is fixed-base; only the pendant demo is meaningful
            s = run_demo(r, v, out_dir, n_seconds=args.seconds, fps=args.fps)
            summaries.append(s)
            verdict = "STABLE" if s["stable"] else "UNSTABLE"
            print(
                f"[stand_demo] {r:>3s} {v:>9s}  "
                f"z {s['root_z_initial']:.3f}→{s['root_z_final']:.3f}  "
                f"max|qvel|@end={s['max_qvel_final_quarter']:6.2f}  "
                f"{verdict}  -> {s.get('mp4', '(no mp4)')}"
            )

    (out_dir / "stand_summary.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
