"""Run an mjlab task with a trained policy, record ctrl trajectory + render mp4.

The recorded ctrl sequence is then replayed in pure MuJoCo (the MetaSim path)
by `replay_in_metasim.py`, producing the right-hand half of the side-by-side.

Output dir layout:
    <out>/
      mjlab_native.mp4      — left-side video, rendered via mjlab env.render()
      trajectory.npz        — ctrl/qpos/qvel/time per control step

The npz schema matches `RolloutResult` so existing tooling can ingest it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help='mjlab task id, e.g. "Mjlab-Tracking-Flat-Unitree-G1"')
    p.add_argument("--checkpoint", required=True, help="path to .pt RSL-RL checkpoint")
    p.add_argument("--motion", default=None, help="path to LAFAN motion npz (tracking tasks only)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--device", default=None)
    p.add_argument("--render-every", type=int, default=1, help="capture frame every N control steps")
    p.add_argument(
        "--nconmax",
        type=int,
        default=None,
        help="Override mujoco-warp nconmax (rough-terrain tasks need ≥256). Default uses task cfg.",
    )
    p.add_argument("--njmax", type=int, default=None, help="Override mujoco-warp njmax. Default uses task cfg.")
    args = p.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")

    # mjlab imports (deferred — heavy)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_idx = 0

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    if args.motion is not None:
        try:
            from mjlab.tasks.tracking.mdp import MotionCommandCfg

            if "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg):
                env_cfg.commands["motion"].motion_file = str(args.motion)
                print(f"[mjlab_policy] motion: {args.motion}")
        except ImportError:
            pass  # tracking module may not be available for non-tracking tasks

    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.height = args.height
        env_cfg.viewer.width = args.width

    # mujoco-warp's nconmax / njmax must fit the actual contact count at runtime.
    # Rough terrain tasks blow past the default. Lives on SimulationCfg (env_cfg.sim).
    if args.nconmax is not None and hasattr(env_cfg.sim, "nconmax"):
        env_cfg.sim.nconmax = int(args.nconmax)
        print(f"[mjlab_policy] override nconmax={args.nconmax}")
    if args.njmax is not None and hasattr(env_cfg.sim, "njmax"):
        env_cfg.sim.njmax = int(args.njmax)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    base_env = env  # for .sim access before wrapping
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[mjlab_policy] loaded checkpoint: {args.checkpoint}")

    # Probe shapes
    sim_data = base_env.sim.data
    nq = int(sim_data.qpos.shape[1])
    nv = int(sim_data.qvel.shape[1])
    nu = int(sim_data.ctrl.shape[1])
    dt = float(env_cfg.sim.mujoco.timestep)
    decimation = int(env_cfg.decimation)
    print(f"[mjlab_policy] task={args.task} nq={nq} nv={nv} nu={nu} dt={dt} decimation={decimation}")

    qpos_log = [sim_data.qpos[env_idx].cpu().numpy().copy()]
    qvel_log = [sim_data.qvel[env_idx].cpu().numpy().copy()]
    ctrl_log: list[np.ndarray] = []
    time_log = [0.0]
    frames: list[np.ndarray] = []

    obs = env.get_observations()
    t_start = time.time()

    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        step_out = env.step(actions)
        obs = step_out[0]

        # Capture per-step ctrl + state.
        ctrl_log.append(sim_data.ctrl[env_idx].cpu().numpy().copy())
        qpos_log.append(sim_data.qpos[env_idx].cpu().numpy().copy())
        qvel_log.append(sim_data.qvel[env_idx].cpu().numpy().copy())
        time_log.append((step + 1) * dt * decimation)

        if step % max(1, args.render_every) == 0:
            try:
                img = env.unwrapped.render()
                if img is not None:
                    img = np.asarray(img)
                    if img.ndim == 4:
                        img = img[0]
                    frames.append(img)
            except Exception as e:
                if step == 0:
                    print(f"[mjlab_policy] render error: {e}")

        if step % 50 == 0:
            print(
                f"[mjlab_policy] step={step:>4}/{args.steps}  frames={len(frames)}  elapsed={time.time() - t_start:.1f}s"
            )

    env.close()

    # Save trajectory npz + compiled scene.
    import mujoco

    mj_model = base_env.sim.mj_model
    # Dump the exact compiled scene (terrain + robot + actuators) for replay.
    scene_path = out / "compiled_scene.mjb"
    mujoco.mj_saveModel(mj_model, str(scene_path), None)
    print(f"[mjlab_policy] wrote {scene_path}  (njnt={mj_model.njnt} nbody={mj_model.nbody})")
    joint_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt_{i}" for i in range(mj_model.njnt)
    ]
    actuator_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}" for i in range(mj_model.nu)
    ]
    npz_path = out / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        backend=np.array("mjlab_native"),
        task_id=np.array(args.task),
        dt=np.float64(dt),
        decimation=np.int32(decimation),
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log) if ctrl_log else np.zeros((0, nu)),
        time=np.array(time_log),
        joint_names=np.array(joint_names),
        actuator_names=np.array(actuator_names),
    )
    print(f"[mjlab_policy] wrote {npz_path}")

    # Save mp4.
    if frames:
        try:
            import imageio

            ctrl_dt = dt * decimation * max(1, args.render_every)
            fps = max(1, round(1.0 / ctrl_dt))
            mp4_path = out / "mjlab_native.mp4"
            imageio.mimwrite(mp4_path, frames, fps=fps, codec="libx264", quality=7)
            print(f"[mjlab_policy] wrote {mp4_path} ({len(frames)} frames @ {fps} fps)")
        except Exception as e:
            print(f"[mjlab_policy] mp4 write failed: {e}")
    else:
        print("[mjlab_policy] no frames captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
