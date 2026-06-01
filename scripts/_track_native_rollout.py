"""mjlab-native closed-loop rollout of the pretrained G1 tracking policy.

Builds the native ``Mjlab-Tracking-Flat-Unitree-G1`` env in play mode with the
LAFAN motion + the canonical demo checkpoint, runs it deterministically for N
control steps, and dumps base-height, anchor-tracking-error, and the per-step
robot joint_pos / motion ref joint_pos trajectories to an npz for comparison
against the RoboVerse rollout.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES=1 python scripts/_track_native_rollout.py \
        --ckpt /tmp/claude-0/mjlab_cache/demo_ckpt.pt \
        --motion /tmp/claude-0/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
        --steps 400 --out /tmp/claude-0/track_native.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

TASK = "Mjlab-Tracking-Flat-Unitree-G1"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", required=True)
    p.add_argument("--render", default=None)
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.environ.get("MJLAB_REPO", "/workspace/mjlab_upstream"), "src"))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES pins physical GPU

    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = args.motion
    motion_cmd.sampling_mode = "start"  # play deterministic: start at frame 0
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    env_cfg.scene.num_envs = 1

    base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(args.ckpt, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd = base.command_manager.get_term("motion")
    robot = base.scene["robot"]

    sd = base.sim.data

    obs = env.get_observations()

    z_log, anchor_err, body_err = [], [], []
    robot_jpos, ref_jpos, time_idx = [], [], []
    frames = []
    renderer = cam = render_model = render_data = None
    if args.render:
        import mujoco

        # mjlab runs physics in mujoco-warp; copy the env-0 qpos/qvel into a host
        # mjData each step and mj_forward so the classic Renderer can draw it.
        render_model = base.sim.mj_model
        render_data = mujoco.MjData(render_model)
        renderer = mujoco.Renderer(render_model, height=480, width=640)
        bid = mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        if bid < 0:
            bid = mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(render_model, cam)
        cam.distance = 3.0
        cam.elevation = -8.0
        cam.azimuth = 120.0
        render_track_bid = bid if 0 <= bid < render_model.nbody else 0
        print(f"[native] render torso_link bid={bid} nbody={render_model.nbody}")

    for _ in range(args.steps):
        # snapshot reference BEFORE step (reward is scored against current ref)
        a_err = torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1).mean().item()
        b_err = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean().item()
        rjp = cmd.robot_joint_pos[0].detach().cpu().numpy().copy()
        rfp = cmd.joint_pos[0].detach().cpu().numpy().copy()
        ti = int(cmd.time_steps[0].item())

        with torch.inference_mode():
            actions = policy(obs)
        obs = env.step(actions)[0]

        z = float(sd.qpos[0, 2].item())
        z_log.append(z)
        anchor_err.append(a_err)
        body_err.append(b_err)
        robot_jpos.append(rjp)
        ref_jpos.append(rfp)
        time_idx.append(ti)

        if renderer is not None:
            import mujoco

            render_data.qpos[:] = np.asarray(sd.qpos[0].detach().cpu().numpy())
            render_data.qvel[:] = np.asarray(sd.qvel[0].detach().cpu().numpy())
            mujoco.mj_forward(render_model, render_data)
            cam.lookat[:] = render_data.xpos[render_track_bid]
            renderer.update_scene(render_data, camera=cam)
            frames.append(renderer.render().copy())
        if z < 0.2:
            print(f"[native] collapsed at step {len(z_log)} z={z:.3f}")
            break

    z_log = np.array(z_log)
    anchor_err = np.array(anchor_err)
    body_err = np.array(body_err)
    robot_jpos = np.array(robot_jpos)
    ref_jpos = np.array(ref_jpos)
    time_idx = np.array(time_idx)
    np.savez(
        args.out,
        z=z_log,
        anchor_err=anchor_err,
        body_err=body_err,
        robot_jpos=robot_jpos,
        ref_jpos=ref_jpos,
        time_idx=time_idx,
    )
    print(f"[native] steps={len(z_log)} z: min={z_log.min():.3f} mean={z_log.mean():.3f} final={z_log[-1]:.3f}")
    print(f"[native] anchor_err: mean={anchor_err.mean():.4f} max={anchor_err.max():.4f}")
    print(f"[native] body_err:   mean={body_err.mean():.4f} max={body_err.max():.4f}")
    print(f"[native] wrote {args.out}")

    if args.render and frames:
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(args.render)), exist_ok=True)
        imageio.mimwrite(args.render, frames, fps=50, codec="libx264", quality=7)
        print(f"[native] wrote {args.render} ({len(frames)} frames)")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
