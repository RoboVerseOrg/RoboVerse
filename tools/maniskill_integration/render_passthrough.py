"""Render a ManiSkill3/* env demo through our passthrough.

Random-policy rollout to mp4 — demonstrates passthrough integration works.
For real RL training, use mani_skill's own training scripts; this is a
visual smoke test for the integration.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ghr/projects/RoboVerse/RoboVerse")

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help='e.g. "UnitreeG1Stand-v1"')
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=1)
    args = p.parse_args()

    import gymnasium as gym

    env_id = f"ManiSkill3/{args.task}"
    env = gym.make(env_id, num_envs=args.num_envs, render_mode="rgb_array")
    print(f"[maniskill_passthrough] {env_id}  num_envs={args.num_envs}")
    obs, _ = env.reset(seed=0)
    frames = []
    for t in range(args.steps):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        try:
            img = env.render()
            if img is not None:
                # ManiSkill 3 returns CUDA tensors (batched); pull to numpy
                if hasattr(img, "cpu"):
                    img = img.cpu().numpy()
                else:
                    img = np.asarray(img)
                if img.ndim == 4:
                    img = img[0]
                frames.append(img.astype(np.uint8))
        except Exception as e:
            if t == 0:
                print(f"[maniskill_passthrough] render error: {e}")
    env.close()
    print(f"[maniskill_passthrough] {len(frames)} frames captured")

    if not frames:
        print("[maniskill_passthrough] no frames; aborting")
        return 1
    import imageio

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, codec="libx264", quality=7)
    print(f"[maniskill_passthrough] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
