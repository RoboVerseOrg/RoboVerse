"""Side-by-side 1:1 visualization: native ManiSkill | MetaSim-recipe state | diff.

The 1:1 claim is about *physics state*, not pixels — so both columns are drawn
with ManiSkill's own renderer (identical assets, lighting, camera), isolating the
one variable we are proving matches:

* **Left**  — native ManiSkill rollout (``physx_cpu``, bit-deterministic).
* **Right** — the *same* ManiSkill scene with its state overwritten at every step
  by the state the reproduction recipe (``recipe.build_replica``) computes from
  the identical action sequence.
* **Right-most** — the amplified pixel difference.  Object state is frame-bitwise
  and the robot rides on ~1e-6 (demo-like motion), so the diff panel is near-black
  — that *is* the picture of 1:1.

Usage::

    SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.render_parity \
        --task PickCube-v1 --steps 60 --out runs/PickCube-v1/side_by_side.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import recipe as R
from .parity_native import _controller_targets

# Articulation state layout: root_pose(7) + root_vel(6) + qpos(n) + qvel(n).
_ROOT = 13


def _smooth_actions(steps: int, adim: int, amp: float) -> np.ndarray:
    """A smooth, demo-like action sweep that visibly moves the arm + gripper."""
    t = np.linspace(0, 4 * np.pi, steps)
    phase = np.linspace(0, 1.5, adim)
    acts = amp * np.sin(t[:, None] + phase[None, :])
    acts[:, -1] = np.sin(np.linspace(0, 2 * np.pi, steps))  # open/close gripper
    return acts.astype(np.float32)


def _to_img(fr) -> np.ndarray:
    arr = fr.cpu().numpy() if hasattr(fr, "cpu") else np.asarray(fr)
    return arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])[..., :3].astype(np.uint8)


def _replica_states(cap, actions, actor_names):
    """State trajectory from the raw reproduction recipe (recipe.build_replica)."""
    scene, robot, actor_objs = R.build_replica(cap)
    ajoints = robot.get_active_joints()
    rep_qpos, rep_qvel, rep_actor = [], [], {n: [] for n in actor_names}
    for a in actions:
        q = robot.get_qpos().astype(np.float32).copy()
        target = _controller_targets(q, a)
        for k, j in enumerate(ajoints):
            j.set_drive_target(float(target[k]))
        for _ in range(cap.sim.decimation):
            scene.step()
        rep_qpos.append(robot.get_qpos().copy())
        rep_qvel.append(robot.get_qvel().copy())
        for n in actor_names:
            obj = actor_objs[n]
            comp = next((c for c in obj.get_components() if hasattr(c, "linear_velocity")), None)
            p = obj.get_pose()
            lin = np.asarray(comp.linear_velocity).ravel() if comp is not None else np.zeros(3)
            ang = np.asarray(comp.angular_velocity).ravel() if comp is not None else np.zeros(3)
            rep_actor[n].append(np.concatenate([p.p, p.q, lin, ang]))
    return rep_qpos, rep_qvel, rep_actor


def _shipped_states(task_key, cap, actions, actor_names):
    """State trajectory from the SHIPPED maniskill.<key>_native task (standard handler path)."""
    import copy

    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers tasks
    from metasim.task.registry import get_task_class

    cls = get_task_class(f"maniskill.{task_key}_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    task.reset()
    # Match the native reset state.
    task.handler.object_ids["panda"].set_qpos(cap.robot.qpos.astype(np.float32))
    task.handler.object_ids["panda"].set_qvel(cap.robot.qvel.astype(np.float32))
    import sapien

    for s in cap.actors:
        task.handler.object_ids[s.name].set_pose(sapien.Pose(s.pose_p, s.pose_q))
    robot = task.handler.object_ids["panda"]
    rep_qpos, rep_qvel, rep_actor = [], [], {n: [] for n in actor_names}
    for a in actions:
        task.step(torch.tensor(a).unsqueeze(0))
        rep_qpos.append(np.asarray(robot.get_qpos()).ravel().copy())
        rep_qvel.append(np.asarray(robot.get_qvel()).ravel().copy())
        for n in actor_names:
            obj = task.handler.object_ids[n]
            comp = next((c for c in obj.get_components() if hasattr(c, "linear_velocity")), None)
            p = obj.get_pose()
            lin = np.asarray(comp.linear_velocity).ravel() if comp is not None else np.zeros(3)
            ang = np.asarray(comp.angular_velocity).ravel() if comp is not None else np.zeros(3)
            rep_actor[n].append(np.concatenate([p.p, p.q, lin, ang]))
    task.close()
    return rep_qpos, rep_qvel, rep_actor


def run(task_id: str, steps: int, seed: int, amp: float, out_path: Path, shipped_task_key: str | None = None) -> dict:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    # ---- 1. native rollout: frames + captured scene ------------------------
    env = gym.make(
        task_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
    )
    u = env.unwrapped
    env.reset(seed=seed)
    cap = R.capture_native(env)
    adim = env.action_space.shape[-1]
    actions = _smooth_actions(steps, adim, amp)
    native_frames = []
    for a in actions:
        env.step(torch.tensor(a).unsqueeze(0))
        native_frames.append(_to_img(env.render()))

    # ---- 2. record the MetaSim-computed state per step ---------------------
    actor_names = [s.name for s in cap.actors]
    if shipped_task_key is not None:
        rep_qpos, rep_qvel, rep_actor = _shipped_states(shipped_task_key, cap, actions, actor_names)
    else:
        rep_qpos, rep_qvel, rep_actor = _replica_states(cap, actions, actor_names)

    # ---- 3. render the recipe state inside ManiSkill's own scene -----------
    env.reset(seed=seed)
    base = env.unwrapped.get_state_dict()
    art_key = next(iter(base["articulations"]))
    nq = len(cap.robot.qpos)
    metasim_frames = []
    for i in range(steps):
        sd = env.unwrapped.get_state_dict()
        art = base["articulations"][art_key].clone()
        art[0, _ROOT : _ROOT + nq] = torch.as_tensor(rep_qpos[i], dtype=art.dtype)
        art[0, _ROOT + nq : _ROOT + 2 * nq] = torch.as_tensor(rep_qvel[i], dtype=art.dtype)
        sd["articulations"][art_key] = art
        for n in rep_actor:
            if n in sd["actors"]:
                sd["actors"][n] = torch.as_tensor(rep_actor[n][i], dtype=art.dtype).reshape(1, -1)
        env.unwrapped.set_state_dict(sd)
        metasim_frames.append(_to_img(env.render()))
    env.close()

    # ---- 4. composite [native | metasim | diff×8] --------------------------
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = native_frames[0].shape[0]
    pad = np.zeros((H, 6, 3), np.uint8)
    comp, diffs = [], []
    for nf, mf in zip(native_frames, metasim_frames, strict=False):
        d = np.abs(nf.astype(np.int16) - mf.astype(np.int16))
        diffs.append(float(d.mean()))
        dvis = np.clip(d * 8, 0, 255).astype(np.uint8)
        comp.append(np.concatenate([nf, pad, mf, pad, dvis], axis=1))
    imageio.mimwrite(out_path, comp, fps=20, codec="libx264", quality=8, macro_block_size=1)
    return {"task": task_id, "steps": steps, "mean_pixel_diff": float(np.mean(diffs)), "out": str(out_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--shipped",
        default=None,
        help="drive the shipped maniskill.<key>_native task (e.g. pick_cube) instead of the recipe replica",
    )
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(f"runs/{args.task}/side_by_side.mp4")
    r = run(args.task, args.steps, args.seed, args.amp, out, shipped_task_key=args.shipped)
    print(f"{r['task']}: mean_pixel_diff={r['mean_pixel_diff']:.4f}/255  ->  {r['out']}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    raise SystemExit(main())
