"""Headless runner for mjlab's pretrained G1 tracking policy.

mjlab ships a demo policy + LAFAN motion at
  https://storage.googleapis.com/mjlab_beta/{model_49999.pt,lafan_dance1_subject1.npz}
auto-downloaded by `mjlab.scripts.gcs.ensure_default_{checkpoint,motion}`.

`mjlab.scripts.demo` runs this policy but always opens a blocking viewer
(viser or native), so we can't get a non-interactive mp4 out of it. This
script mirrors the same checkpoint-load + env-build + step loop, runs
for a fixed number of control steps, and writes the rendered episode
to `out/g1_dance.mp4`.

This is the "G1 actually moving" deliverable — the LAFAN motion file
choreographs a slow rotational dance with a few side-steps. Once we
have an actual walking checkpoint (or train one), we can swap it in
via --checkpoint-file.
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

DEFAULT_OUT = Path("/home/ghr/projects/RoboVerse/reports/mjlab_integration/runs/g1_pretrained")


def _wrap_actions(action_shape, agent_cfg):
    clip = float(agent_cfg.clip_actions) if agent_cfg.clip_actions is not None else None
    if clip is None:
        return lambda a: a
    return lambda a: torch.clamp(a, -clip, clip)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        default="Mjlab-Tracking-Flat-Unitree-G1",
        help="mjlab task id; demo checkpoint trains on Tracking-Flat-G1",
    )
    p.add_argument("--checkpoint", default=None, help="path to .pt; defaults to mjlab demo ckpt")
    p.add_argument("--motion", default=None, help="path to motion npz; defaults to mjlab demo motion")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(ensure_default_checkpoint())
    motion = Path(args.motion) if args.motion else Path(ensure_default_motion())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    # Wire the motion file if the task is a tracking task.
    if "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg):
        env_cfg.commands["motion"].motion_file = str(motion)
        print(f"[g1_walk] motion: {motion}")

    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.height = args.height
        env_cfg.viewer.width = args.width

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[g1_walk] loaded checkpoint: {checkpoint}")

    obs = env.get_observations()
    frames: list[np.ndarray] = []
    t_start = time.time()

    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        step_out = env.step(actions)
        # rsl_rl wrapper returns (obs, rew, dones, infos); ManagerBasedRlEnv
        # without wrap returns (obs, rew, term, trunc, info). We always take obs.
        obs = step_out[0]
        try:
            img = env.unwrapped.render()
        except Exception as e:
            if step == 0:
                print(f"[g1_walk] render error: {e}")
            img = None
        if img is not None:
            img = np.asarray(img)
            if img.ndim == 4:  # batched render
                img = img[0]
            frames.append(img)
        if step % 50 == 0:
            print(f"[g1_walk] step={step:>4}/{args.steps}  frames={len(frames)}  elapsed={time.time() - t_start:.1f}s")

    env.close()

    if not frames:
        print("[g1_walk] no frames captured — render returned None")
        return 1

    try:
        import imageio

        ctrl_dt = float(getattr(env_cfg.sim, "control_dt", 0.02))
        fps = round(1.0 / max(ctrl_dt, 1e-3))
        out_mp4 = out_dir / "g1_dance.mp4"
        imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", quality=7)
        print(f"[g1_walk] wrote {out_mp4} ({len(frames)} frames @ {fps} fps)")
    except Exception as e:
        print(f"[g1_walk] mp4 write failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
