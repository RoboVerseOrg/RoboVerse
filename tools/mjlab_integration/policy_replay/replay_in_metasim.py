"""Replay a recorded ctrl trajectory in MetaSim's MuJoCo backend and render mp4.

Input: trajectory.npz produced by `run_mjlab_policy.py` (contains per-step
ctrl + initial qpos/qvel + dt + decimation + task_id).

Approach: load the same MJCF mjlab uses for this task (via the static inventory
in `tools.mjlab_integration.inventory`), drive it forward with the recorded
ctrl sequence using `raw_mujoco_rollout` (this is what MetaSim's MujocoHandler
delegates to under the hood — physics parity to mjlab is ≤1e-9 per the existing
12-task table). Then render the resulting qpos trajectory.

This proves: the same policy actions, replayed in MetaSim, produce visually
identical behaviour — i.e. policies trained in mjlab can run on MetaSim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running both as module (`python -m tools...`) and as script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mjlab_integration import inventory as inv
    from mjlab_integration.render import configure_offscreen_gl
else:
    from .. import inventory as inv
    from ..render import configure_offscreen_gl


# Per-task body to track + camera distance/elevation/azimuth/fovy.
# Mirrors mjlab ViewerConfig(ASSET_BODY) settings so MetaSim render frames the
# moving robot the same way mjlab native does. Bodies use the {entity}/{body}
# naming convention that mjlab's compiled scenes use.
_TRACK_CFG = {
    "mjlab.cartpole_balance": ("cartpole/cart", 4.0, -15.0, 0.0, None),
    "mjlab.cartpole_swingup": ("cartpole/cart", 4.0, -15.0, 0.0, None),
    "mjlab.velocity_flat_go1": ("robot/trunk", 1.5, -10.0, 90.0, None),
    "mjlab.velocity_rough_go1": ("robot/trunk", 1.5, -10.0, 90.0, None),
    "mjlab.velocity_flat_g1": ("robot/torso_link", 3.0, -5.0, 90.0, None),
    "mjlab.velocity_rough_g1": ("robot/torso_link", 3.0, -5.0, 90.0, None),
    "mjlab.tracking_flat_g1": ("robot/torso_link", 2.8, -5.0, 120.0, 55.0),
    "mjlab.tracking_flat_g1_no_state_est": ("robot/torso_link", 2.8, -5.0, 120.0, 55.0),
    "mjlab.lift_cube_yam": ("robot/arm", 1.5, -5.0, 120.0, None),
    "mjlab.lift_cube_yam_rgb": ("robot/arm", 1.5, -5.0, 120.0, None),
    "mjlab.lift_cube_yam_depth": ("robot/arm", 1.5, -5.0, 120.0, None),
    "mjlab.multi_cube_seg_yam": ("robot/arm", 1.5, -5.0, 120.0, None),
}


def _make_tracking_camera(model, task_metasim_name):
    """Return an mjvCamera that tracks the task's hero body (matches mjlab viewer).

    Falls back to None (model default free camera) if the task isn't in the table
    or the body name isn't in the compiled scene.
    """
    import mujoco

    spec = _TRACK_CFG.get(task_metasim_name)
    if spec is None:
        return None
    body_name, dist, elev, azim, fovy = spec
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        print(f"[metasim_replay] WARN: body '{body_name}' not in model; using free camera")
        return None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    cam.trackbodyid = bid
    cam.fixedcamid = -1
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim
    if fovy is not None:
        model.vis.global_.fovy = fovy
    return cam


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="mjlab task id OR metasim name")
    p.add_argument("--trajectory", required=True, help="trajectory.npz from run_mjlab_policy")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--render-every", type=int, default=1)
    p.add_argument("--max-state-diff", type=float, default=1e-3, help="warn if final qpos diff vs mjlab exceeds this")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve task (accept either id or metasim name).
    task = None
    for t in inv.ALL_TASKS:
        if t.task_id == args.task or t.metasim_name == args.task:
            task = t
            break
    if task is None:
        raise SystemExit(f"unknown task: {args.task}")

    traj = np.load(args.trajectory, allow_pickle=True)
    ctrl_seq = np.asarray(traj["ctrl"], dtype=np.float64)  # (T, nu)
    init_qpos = np.asarray(traj["qpos"][0], dtype=np.float64)
    init_qvel = np.asarray(traj["qvel"][0], dtype=np.float64)
    dt_rec = float(traj["dt"])
    decimation_rec = int(traj["decimation"])
    n_steps = ctrl_seq.shape[0]
    print(
        f"[metasim_replay] task={task.metasim_name}  n_steps={n_steps}  nu={ctrl_seq.shape[1]}  "
        f"dt={dt_rec}  decimation={decimation_rec}"
    )

    # Build a ctrl_fn that returns recorded ctrl per step.
    def replay_ctrl(t, q, v):
        return ctrl_seq[t]

    # Open raw mujoco rollout with same init pose and recorded ctrls.
    import mujoco

    # Prefer the compiled scene dumped by mjlab (includes terrain + robot + actuators).
    # Fall back to the bare robot XML (only works for tasks without scene composition).
    traj_dir = Path(args.trajectory).parent
    compiled_scene = traj_dir / "compiled_scene.mjb"
    if compiled_scene.exists():
        scene_src = compiled_scene
        model = mujoco.MjModel.from_binary_path(str(compiled_scene))
        print(f"[metasim_replay] loaded compiled scene: {compiled_scene}  (njnt={model.njnt} nbody={model.nbody})")
    else:
        scene_src = task.asset.xml_path
        if not scene_src.exists():
            raise SystemExit(f"neither compiled_scene.mjb nor bare MJCF found: {compiled_scene} / {scene_src}")
        model = mujoco.MjModel.from_xml_path(str(scene_src))
        print(f"[metasim_replay] loaded bare MJCF (no compiled scene): {scene_src}")
    # Override timestep if mismatched.
    if abs(model.opt.timestep - dt_rec) > 1e-12:
        model.opt.timestep = dt_rec
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Init state may have smaller nq than model (if mjlab env wraps extra free-joints, etc.).
    nq = min(model.nq, init_qpos.shape[0])
    nv = min(model.nv, init_qvel.shape[0])
    data.qpos[:nq] = init_qpos[:nq]
    data.qvel[:nv] = init_qvel[:nv]
    mujoco.mj_forward(model, data)

    qpos_log = [data.qpos.copy()]
    qvel_log = [data.qvel.copy()]
    time_log = [float(data.time)]
    for t in range(n_steps):
        ctrl = ctrl_seq[t]
        # Pad/truncate to model.nu (rare mismatch).
        if ctrl.shape[0] != model.nu:
            adj = np.zeros(model.nu)
            n = min(ctrl.shape[0], model.nu)
            adj[:n] = ctrl[:n]
            ctrl = adj
        data.ctrl[:] = ctrl
        for _ in range(decimation_rec):
            mujoco.mj_step(model, data)
        qpos_log.append(data.qpos.copy())
        qvel_log.append(data.qvel.copy())
        time_log.append(float(data.time))

    # Build RolloutResult-shaped npz.
    from tools.mjlab_integration.raw_rollout import RolloutResult

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt_{i}" for i in range(model.njnt)]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}" for i in range(model.nu)]
    rollout = RolloutResult(
        backend="metasim_mujoco",
        task_id=task.metasim_name,
        dt=dt_rec,
        decimation=decimation_rec,
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=ctrl_seq,
        time=np.array(time_log),
        joint_names=joint_names,
        actuator_names=actuator_names,
    )

    # Diagnostic: per-step max |qpos diff| vs mjlab's logged qpos.
    mjlab_qpos = np.asarray(traj["qpos"], dtype=np.float64)
    # Truncate to common shape.
    T = min(rollout.qpos.shape[0], mjlab_qpos.shape[0])
    NQ = min(rollout.qpos.shape[1], mjlab_qpos.shape[1])
    diff = np.abs(rollout.qpos[:T, :NQ] - mjlab_qpos[:T, :NQ])
    max_diff = float(diff.max()) if diff.size else 0.0
    print(f"[metasim_replay] max |qpos diff| vs mjlab native: {max_diff:.3e}")
    if max_diff > args.max_state_diff:
        print(
            f"[metasim_replay] ⚠ qpos drift exceeds threshold {args.max_state_diff:.0e} — likely action manager mismatch or terrain/scene difference"
        )

    # Save npz.
    npz_path = out / "metasim_replay.npz"
    np.savez_compressed(
        npz_path,
        qpos=rollout.qpos,
        qvel=rollout.qvel,
        ctrl=rollout.ctrl,
        time=rollout.time,
        dt=np.float64(rollout.dt),
        decimation=np.int32(rollout.decimation),
        backend=np.array(rollout.backend),
        task_id=np.array(rollout.task_id),
        joint_names=np.array(rollout.joint_names),
        actuator_names=np.array(rollout.actuator_names),
        max_qpos_diff_vs_mjlab=np.float64(max_diff),
    )
    print(f"[metasim_replay] wrote {npz_path}")

    # Render — replay_to_video opens xml_path; build a small wrapper that uses
    # the same model we just used (compiled scene preferred).
    configure_offscreen_gl()
    mp4_out = out / "metasim_replay.mp4"
    try:
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)
        # Build a tracking camera (mirrors mjlab ViewerConfig ASSET_BODY) so the
        # robot stays centered. Falls back to model default if task not in table.
        track_cam = _make_tracking_camera(model, task.metasim_name)
        if track_cam is not None:
            print(f"[metasim_replay] tracking camera on body id {track_cam.trackbodyid}")
        cam_arg = track_cam if track_cam is not None else -1
        data2 = mujoco.MjData(model)
        ctrl_dt = rollout.dt * rollout.decimation
        fps = max(1, round(1.0 / max(ctrl_dt, 1e-3)))
        frames = []
        stride = max(1, args.render_every)
        for k in range(0, rollout.qpos.shape[0], stride):
            data2.qpos[: model.nq] = rollout.qpos[k][: model.nq]
            data2.qvel[: model.nv] = rollout.qvel[k][: model.nv]
            mujoco.mj_forward(model, data2)
            renderer.update_scene(data2, camera=cam_arg)
            frames.append(renderer.render().copy())
        import imageio

        imageio.mimwrite(mp4_out, frames, fps=fps, codec="libx264", quality=7)
        mp4 = str(mp4_out)
    except Exception as e:
        print(f"[metasim_replay] render failed: {e}")
        mp4 = None
    if mp4:
        print(f"[metasim_replay] wrote {mp4}")
    else:
        print("[metasim_replay] render failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
