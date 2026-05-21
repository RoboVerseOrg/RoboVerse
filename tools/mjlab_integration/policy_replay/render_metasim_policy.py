"""Roll out a MetaSim-trained policy in our MetaSim env, render mp4.

The "true" RL-trained-in-our-framework demo: load our PPO checkpoint, step
through MetaSim env, capture frames with the same tracking camera used for
mjlab side-by-side. Output is rendered through MetaSim's MujocoHandler — not
mjlab runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/ghr/projects/RoboVerse/RoboVerse")

import roboverse_pack  # noqa: F401  register tasks
from metasim.task.registry import get_task_class
from tools.mjlab_integration.policy_replay.train_metasim_cartpole import Agent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="mjlab.cartpole_balance_train")
    p.add_argument("--ckpt", required=True, help="path to agent_final.pt from train_metasim_cartpole")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls = get_task_class(args.task)
    env = cls()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    print(f"[metasim_policy] task={args.task} obs={obs_dim} act={act_dim}")

    # Load our trained PPO agent
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = state.get("args", {})
    agent = Agent(
        obs_dim,
        act_dim,
        activation=saved_args.get("activation", "elu"),
        init_std=saved_args.get("init_std", 1.0),
    ).to(device)
    agent.load_state_dict(state["agent"])
    agent.eval()
    print(f"[metasim_policy] loaded MetaSim-native PPO ckpt: {args.ckpt}")

    # Render setup — tracking camera. Our MetaSim scene loads the bare MJCF
    # (no entity namespace), so bodies are "cart" / "pole_1", not "cartpole/cart".
    import mujoco

    model = env.handler.physics.model.ptr if hasattr(env.handler.physics.model, "ptr") else env.handler.physics.model
    # Try a known list of body names for this task family
    track_candidates = {
        "mjlab.cartpole_balance_train": ("cart", 4.0, -15.0, 0.0),
        "mjlab.cartpole_swingup_train": ("cart", 4.0, -15.0, 0.0),
    }
    cam = None
    spec = track_candidates.get(args.task)
    if spec is not None:
        body_name, dist, elev, azim = spec
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
            cam.trackbodyid = bid
            cam.fixedcamid = -1
            cam.distance = dist
            cam.elevation = elev
            cam.azimuth = azim
            print(f"[metasim_policy] tracking camera on body '{body_name}' (id {bid})")
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)

    obs, _ = env.reset()
    frames = []
    ep_r = 0.0
    for t in range(args.steps):
        with torch.no_grad():
            mean = agent.actor_mean(obs.to(device))  # deterministic eval (no sampling)
        action_np = mean.cpu().numpy().flatten()
        obs, r, term, tout, _ = env.step(action_np)
        ep_r += float(r.item())
        # Render current state via separate mjData snapshot so we don't disturb the env
        renderer.update_scene(
            env.handler.physics.data._data if hasattr(env.handler.physics.data, "_data") else env.handler.physics.data,
            camera=cam if cam is not None else -1,
        )
        frames.append(renderer.render().copy())
        if term.any() or tout.any():
            print(f"[metasim_policy] episode ended at step {t + 1}, ep_r={ep_r:.3f}")
            break

    print(f"[metasim_policy] {len(frames)} frames, ep_r={ep_r:.3f}")

    # Write mp4
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    # control_dt = dt * decimation = 0.01 * 5 = 0.05 → 20 fps
    fps = 20
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=7)
    print(f"[metasim_policy] wrote {out_path} ({len(frames)} frames @ {fps} fps)")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
