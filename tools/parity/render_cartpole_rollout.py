"""Render a cartpole_balance_v2 rollout to mp4 using mujoco viewer.

Usage:
  python tools/parity/render_cartpole_rollout.py --ckpt path/to/model_200.pt --out /tmp/cartpole.mp4 --steps 400
"""

from __future__ import annotations

import argparse
import os

import imageio
import mujoco
import torch
import torch.nn as nn

import roboverse_pack  # noqa: F401
from metasim.task.registry import get_task_class


def build_actor(state_dict, device):
    keys = sorted(
        [k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    mods = []
    for i, k in enumerate(keys):
        w = state_dict[k]
        mods.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(keys) - 1:
            mods.append(nn.ELU())
    actor = nn.Sequential(*mods).to(device)
    actor.load_state_dict(
        {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")},
        strict=False,
    )
    actor.eval()
    return actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--task", default="mjlab.cartpole_balance_v2")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = build_actor(ckpt["actor_state_dict"], device)
    print(f"Loaded actor from {args.ckpt}")

    # Use mujoco scene-MJCF path for rendering (Newton GPU has no built-in render)
    env = get_task_class(args.task)(device=device)
    env.reset()
    print(f"Env reset; obs.shape = {env._obs_buf['actor'].shape}")

    # Get the physics model for rendering
    physics = env.handler.physics
    model = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    data = physics.data.ptr if hasattr(physics.data, "ptr") else physics.data

    # Renderer setup
    os.environ.setdefault("MUJOCO_GL", "egl")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    frames = []
    rewards = []
    for i in range(args.steps):
        with torch.no_grad():
            a = actor(env._obs_buf["actor"])
        _, r, term, trunc, _ = env.step(a)
        rewards.append(r.item())

        # Update renderer (uses the physics.data we already have)
        renderer.update_scene(data, camera=0 if model.ncam > 0 else -1)
        frame = renderer.render()
        frames.append(frame)

        if term.any():
            print(f"Terminated at step {i}: term={term.item()} trunc={trunc.item()}")
            # Don't break — let env reset and continue

    print(f"Captured {len(frames)} frames, mean reward = {sum(rewards) / len(rewards):.4f}")
    # Write mp4
    fps = 1.0 / env.step_dt if env.step_dt > 0 else 50
    imageio.mimwrite(args.out, frames, fps=int(fps), codec="libx264", quality=8)
    print(f"Wrote {args.out} ({len(frames)} frames @ {fps:.0f} fps, {len(frames) / fps:.1f}s)")


if __name__ == "__main__":
    main()
