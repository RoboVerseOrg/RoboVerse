"""Side-by-side **demo action-replay** video: native ManiSkill | shipped native task | diff.

Unlike :mod:`render_parity` (which drives a synthetic action sweep to probe free-space dynamics),
this tool replays an **official ManiSkill demonstration** — the recorded ``pd_joint_delta_pos``
action sequence — through both native ManiSkill and the shipped ``maniskill.<key>_native`` task,
**from the same demo initial state**, and renders the result. It selects an episode where *both*
sides reach the task's success predicate, so the clip is a genuine end-to-end **1:1 task
completion**: left and right both grasp/place the object, and the diff panel stays near-black.

Both columns are rendered with ManiSkill's renderer (identical assets/lighting/camera). The left
column is native ManiSkill's own rollout; the middle column overwrites that same scene, every step,
with the state the shipped MetaSim task computed from the identical actions — so the only variable
on screen is the physics state, which the friction-corrected recipe now tracks to ~2e-4.

Usage::

    SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.render_demo_replay \
        --task PickCube-v1 --shipped pick_cube \
        --demo ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
        --out runs/demo/PickCube-v1.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Articulation env-state layout: root_pose(7) + root_vel(6) + qpos(n) + qvel(n).
_ROOT = 13


def _to_img(fr) -> np.ndarray:
    arr = fr.cpu().numpy() if hasattr(fr, "cpu") else np.asarray(fr)
    return arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])[..., :3].astype(np.uint8)


def _demo_initial_state_dict(g):
    """Reconstruct ManiSkill ``set_state_dict`` input from a demo episode's first env_state."""
    import torch

    sd = {"actors": {}, "articulations": {}}
    for n in g["env_states"]["actors"]:
        sd["actors"][n] = torch.as_tensor(np.asarray(g["env_states"]["actors"][n])[0], dtype=torch.float32).reshape(
            1, -1
        )
    for n in g["env_states"]["articulations"]:
        sd["articulations"][n] = torch.as_tensor(
            np.asarray(g["env_states"]["articulations"][n])[0], dtype=torch.float32
        ).reshape(1, -1)
    return sd


def _native_action(env, a):
    """Format a flat demo action for the native env (a tensor, or a per-agent dict for multi-agent)."""
    import gymnasium as gym
    import torch

    space = env.action_space
    if isinstance(space, gym.spaces.Dict):
        out, off = {}, 0
        for key, sub in space.spaces.items():
            n = int(np.prod(sub.shape))
            out[key] = torch.tensor(a[off : off + n], dtype=torch.float32).unsqueeze(0)
            off += n
        return out
    return torch.tensor(a).unsqueeze(0)


def _native_rollout(env, g, render: bool, track: str | None = None):
    """Replay a demo episode in the native env from its initial state.

    Returns ``(frames, success_step, track_positions)`` — ``success_step`` is the index of the first
    step at which the task's success predicate fires (``None`` if it never does), used both as the
    both-success gate and to truncate the clip at task completion. ``track_positions`` is the per-step
    world position of actor ``track`` (for native-vs-shipped agreement ranking) or ``None``.
    """
    u = env.unwrapped
    env.reset(seed=0)
    u.set_state_dict(_demo_initial_state_dict(g))
    actions = np.asarray(g["actions"])
    frames, success_step, track_pos = [], None, ([] if track else None)
    for i, a in enumerate(actions):
        _, _, _, _, info = env.step(_native_action(env, a))
        if success_step is None and bool(np.asarray(info["success"]).reshape(-1)[0]):
            success_step = i
        if render:
            frames.append(_to_img(env.render()))
        if track is not None:
            track_pos.append(np.asarray(u.get_state_dict()["actors"][track]).ravel()[:3].copy())
    return frames, success_step, track_pos


def _demo_robot_key(g) -> str:
    """The demo's single robot articulation key (``panda`` / ``panda_wristcam`` / …)."""
    return next(iter(g["env_states"]["articulations"].keys()))


def _robot_map(task, g):
    """Pair each demo robot articulation key with the shipped task's robot name (order-aligned).

    Single-robot: ``[(demo_key, "panda")]``. Multi-robot (TwoRobotPickCube): the demo's
    ``panda_wristcam-agent-{0,1}`` keys map onto the task's ``robot_names`` (``panda-0``, ``panda-1``).
    """
    demo_keys = list(g["env_states"]["articulations"].keys())
    shipped_names = list(getattr(task, "robot_names", None) or [getattr(task, "robot_name", "panda")])
    if len(demo_keys) != len(shipped_names):
        raise RuntimeError(f"robot count mismatch: demo {demo_keys} vs shipped {shipped_names}")
    return list(zip(demo_keys, shipped_names))


def _shipped_rollout(task, g, goal_actor, scene_objs, record_state: bool, actor_remap=None, track: str | None = None):
    """Replay a demo episode through the shipped task from the demo initial state (1 or N robots).

    Returns ``(states, success, track_positions)`` where each ``states`` entry is
    ``({demo_robot_key: (qpos, qvel)}, {actor: 13-vec})`` — generalized over the robot count so the
    same path renders single-robot and multi-robot (TwoRobotPickCube) tasks. ``actor_remap`` maps a
    demo actor name to the shipped object name when they differ (e.g. ``box_with_hole`` →
    ``box_with_hole_0``); recorded states stay keyed by the *demo* name so they inject back into
    ManiSkill's own scene. ``track_positions`` is the per-step world position of demo actor ``track``
    (for agreement ranking) or ``None``.
    """
    import sapien
    import sapien.physx as physx
    import torch

    actor_remap = actor_remap or {}
    task.reset()
    robot_map = _robot_map(task, g)
    for demo_key, rname in robot_map:
        art = np.asarray(g["env_states"]["articulations"][demo_key])[0]
        nq = (len(art) - _ROOT) // 2
        rb = task.handler.object_ids[rname]
        rb.set_qpos(art[_ROOT : _ROOT + nq].astype(np.float32))
        rb.set_qvel(art[_ROOT + nq : _ROOT + 2 * nq].astype(np.float32))

    def shipped_obj(demo_name):
        sname = actor_remap.get(demo_name, demo_name)
        return sname if sname in scene_objs and "table" not in sname else None

    for name in g["env_states"]["actors"]:
        sname = shipped_obj(name)
        if sname is not None:
            # Set the FULL demo state — pose AND velocity. Setting only the pose leaves the previous
            # rollout's residual velocity on the body (set_pose does not zero it), which a chaotic
            # task (rolling ball) amplifies into a different outcome on the next replay.
            p = np.asarray(g["env_states"]["actors"][name])[0]
            obj = task.handler.object_ids[sname]
            obj.set_pose(sapien.Pose(p[:3], p[3:7]))
            comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if comp is not None and len(p) >= _ROOT:
                comp.set_linear_velocity(p[7:10].astype(np.float32))
                comp.set_angular_velocity(p[10:13].astype(np.float32))
    if goal_actor is not None:
        task.goal_pos = np.asarray(g["env_states"]["actors"][goal_actor])[0][:3].astype(np.float64)

    actor_pairs = [(n, shipped_obj(n)) for n in g["env_states"]["actors"]]
    actor_pairs = [(demo_n, ship_n) for demo_n, ship_n in actor_pairs if ship_n is not None]
    track_ship = shipped_obj(track) if track is not None else None
    states, success, track_pos = [], False, ([] if track is not None else None)
    for a in np.asarray(g["actions"]):
        out = task.step(torch.tensor(a, dtype=torch.float32).unsqueeze(0))
        success = success or bool(out[2].reshape(-1)[0])
        if track_ship is not None:
            track_pos.append(np.asarray(task.handler.object_ids[track_ship].get_pose().p).ravel().copy())
        if record_state:
            robot_states = {}
            for demo_key, rname in robot_map:
                rb = task.handler.object_ids[rname]
                robot_states[demo_key] = (
                    np.asarray(rb.get_qpos()).ravel().copy(),
                    np.asarray(rb.get_qvel()).ravel().copy(),
                )
            actor_states = {}
            for demo_n, ship_n in actor_pairs:
                obj = task.handler.object_ids[ship_n]
                comp = next((c for c in obj.get_components() if hasattr(c, "linear_velocity")), None)
                pose = obj.get_pose()
                lin = np.asarray(comp.linear_velocity).ravel() if comp is not None else np.zeros(3)
                ang = np.asarray(comp.angular_velocity).ravel() if comp is not None else np.zeros(3)
                actor_states[demo_n] = np.concatenate([pose.p, pose.q, lin, ang])
            states.append((robot_states, actor_states))
    return states, success, track_pos


def _primary_manipuland(g, scene_objs) -> str | None:
    """Name of the most-moved task object over the demo (the thing the task acts on)."""
    best_name, best = None, 0.0
    for name in g["env_states"]["actors"]:
        if name not in scene_objs or "table" in name or "goal" in name:
            continue
        traj = np.asarray(g["env_states"]["actors"][name])[:, :3]
        d = float(np.linalg.norm(traj - traj[0], axis=1).max())
        if d >= best:
            best_name, best = name, d
    return best_name


def _manipuland_motion(g, scene_objs) -> float:
    """Peak displacement of the most-moved task object over the demo (visual-salience score)."""
    name = _primary_manipuland(g, scene_objs)
    if name is None:
        return 0.0
    traj = np.asarray(g["env_states"]["actors"][name])[:, :3]
    return float(np.linalg.norm(traj - traj[0], axis=1).max())


def _rank_episodes(
    env, task, traj_keys, f, goal_actor, scene_objs, max_scan, actor_remap=None, rank_by="agreement", truncate_buffer=8
):
    """Rank demo-success episodes where BOTH native and shipped reach success.

    Scans the first ``max_scan`` demo-success episodes, keeps those where native *and* shipped both
    reach the task's success predicate under pure action replay, and orders the survivors by:

    * ``"agreement"`` (default) — *lowest* native-vs-shipped manipuland trajectory divergence first,
      i.e. the cleanest 1:1 clip. For contact-sensitive tasks (PushT, RollBall) the most aggressive
      push tracks worst, so picking the tightest-tracking completion gives an honestly cleaner video.
    * ``"motion"`` — *largest* manipuland displacement first (most visually dramatic).

    The caller render-verifies the survivors in order, which also guards chaotic tasks where a
    marginal episode can sit on the success boundary.
    """
    candidates = []
    for tk in traj_keys[:max_scan]:
        g = f[tk]
        if not bool(np.asarray(g["success"])[-1]):
            continue
        obj = _primary_manipuland(g, scene_objs)
        _, nat_step, nat_track = _native_rollout(env, g, render=False, track=obj)
        if nat_step is None:
            continue
        _, shp_ok, shp_track = _shipped_rollout(
            task, g, goal_actor, scene_objs, record_state=False, actor_remap=actor_remap, track=obj
        )
        if not shp_ok:
            continue
        if rank_by == "agreement" and nat_track and shp_track:
            # Score the divergence only over the *shown* window — up to task completion (+buffer) —
            # since the clip is truncated there. A demo that keeps fidgeting the manipuland for dozens
            # of post-success steps must not be penalised for drift the viewer never sees.
            n = min(len(nat_track), len(shp_track), nat_step + 1 + truncate_buffer)
            divergence = float(np.linalg.norm(np.asarray(nat_track[:n]) - np.asarray(shp_track[:n]), axis=1).max())
            candidates.append((divergence, tk))  # ascending → tightest first
        else:
            candidates.append((-_manipuland_motion(g, scene_objs), tk))  # ascending of negative → largest first
    return [tk for _, tk in sorted(candidates, key=lambda c: c[0])]


def run(
    task_id,
    shipped_key,
    demo_path,
    out_path,
    goal_actor=None,
    max_scan=40,
    episode=None,
    render_attempts=4,
    actor_remap=None,
    rank_by="agreement",
    truncate_buffer=8,
) -> dict:
    import copy

    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import mani_skill.envs  # noqa: F401
    import torch

    import roboverse_pack.tasks.maniskill  # noqa: F401 — registers tasks
    from metasim.task.registry import get_task_class

    f = h5py.File(os.path.expanduser(demo_path), "r")
    traj_keys = sorted((k for k in f if k.startswith("traj_")), key=lambda s: int(s.split("_")[1]))

    env = gym.make(
        task_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
    )
    cls = get_task_class(f"maniskill.{shipped_key}_native")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    task = cls(sc)
    scene_objs = {o.name for o in sc.objects}

    candidates = (
        [episode]
        if episode
        else _rank_episodes(
            env,
            task,
            traj_keys,
            f,
            goal_actor,
            scene_objs,
            max_scan,
            actor_remap=actor_remap,
            rank_by=rank_by,
            truncate_buffer=truncate_buffer,
        )
    )
    if not candidates:
        env.close()
        task.close()
        f.close()
        raise RuntimeError(f"{task_id}: no episode reached success on BOTH native and shipped in first {max_scan}")

    # Render-verify the ranked candidates: keep the first whose *rendered* rollout has both native and
    # shipped reach success. This guards chaotic tasks where a marginal episode passes a no-render
    # pass but lands the manipuland just outside the goal on the recorded rollout.
    chosen = None
    for tk in candidates[:render_attempts]:
        g = f[tk]
        native_frames, nat_step, _ = _native_rollout(env, g, render=True)
        shipped_states, shp_ok, _ = _shipped_rollout(
            task, g, goal_actor, scene_objs, record_state=True, actor_remap=actor_remap
        )
        if nat_step is not None and shp_ok:
            chosen = (tk, native_frames, shipped_states, nat_step)
            break
    if chosen is None:
        env.close()
        task.close()
        f.close()
        raise RuntimeError(f"{task_id}: no candidate confirmed both-success on render in {render_attempts} attempts")
    tk, native_frames, shipped_states, nat_step = chosen
    nat_ok = shp_ok = True
    g = f[tk]
    task.close()

    # Truncate the clip at task completion (+buffer): the demo keeps running long after success, and
    # those post-success steps are where a contact-rich task (PushT) accumulates drift the viewer
    # never needs to see. The shown window is the actual 1:1 completion.
    if truncate_buffer is not None and nat_step is not None:
        cut = min(len(native_frames), nat_step + 1 + truncate_buffer)
        native_frames = native_frames[:cut]
        shipped_states = shipped_states[:cut]

    # render the shipped state inside ManiSkill's own scene for a pixel-identical comparison
    env.reset(seed=0)
    env.unwrapped.set_state_dict(_demo_initial_state_dict(g))
    base = env.unwrapped.get_state_dict()
    nq_by_key = {
        k: (len(np.asarray(g["env_states"]["articulations"][k])[0]) - _ROOT) // 2
        for k in g["env_states"]["articulations"]
    }
    shipped_frames = []
    for robot_states, actor_states in shipped_states:
        sd = env.unwrapped.get_state_dict()
        for demo_key, (qpos, qvel) in robot_states.items():
            nq = nq_by_key[demo_key]
            art = base["articulations"][demo_key].clone()
            art[0, _ROOT : _ROOT + nq] = torch.as_tensor(qpos, dtype=art.dtype)
            art[0, _ROOT + nq : _ROOT + 2 * nq] = torch.as_tensor(qvel, dtype=art.dtype)
            sd["articulations"][demo_key] = art
        for n, vec in actor_states.items():
            if n in sd["actors"]:
                sd["actors"][n] = torch.as_tensor(vec, dtype=art.dtype).reshape(1, -1)
        env.unwrapped.set_state_dict(sd)
        shipped_frames.append(_to_img(env.render()))
    env.close()
    f.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = native_frames[0].shape[0]
    pad = np.zeros((H, 6, 3), np.uint8)
    comp, diffs = [], []
    for nf, mf in zip(native_frames, shipped_frames):
        d = np.abs(nf.astype(np.int16) - mf.astype(np.int16))
        diffs.append(float(d.mean()))
        dvis = np.clip(d * 8, 0, 255).astype(np.uint8)
        comp.append(np.concatenate([nf, pad, mf, pad, dvis], axis=1))
    imageio.mimwrite(out_path, comp, fps=20, codec="libx264", quality=8, macro_block_size=1)
    return {
        "task": task_id,
        "episode": tk,
        "steps": len(comp),
        "success_step": nat_step,
        "native_success": nat_ok,
        "shipped_success": shp_ok,
        "mean_pixel_diff": float(np.mean(diffs)),
        "out": str(out_path),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--shipped", default="pick_cube")
    ap.add_argument("--demo", required=True, help="path to a pd_joint_delta_pos demo .h5")
    ap.add_argument("--goal-actor", default=None, help="demo actor whose pose seeds task.goal_pos (e.g. goal_site)")
    ap.add_argument("--episode", default=None, help="force a specific traj_<n> instead of auto-selecting")
    ap.add_argument("--max-scan", type=int, default=40)
    ap.add_argument("--render-attempts", type=int, default=4, help="ranked candidates to render-verify")
    ap.add_argument(
        "--actor-remap",
        default=None,
        help="comma-separated demo:shipped actor renames, e.g. box_with_hole:box_with_hole_0,peg:peg_0",
    )
    ap.add_argument(
        "--rank-by",
        default="agreement",
        choices=["agreement", "motion"],
        help="episode pick: 'agreement' = tightest native-vs-shipped tracking (cleanest 1:1); "
        "'motion' = largest manipuland displacement (most dramatic)",
    )
    ap.add_argument(
        "--truncate-buffer",
        type=int,
        default=8,
        help="end the clip this many frames after the task first succeeds (-1 = show the full demo)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(f"runs/demo/{args.task}.mp4")
    remap = dict(p.split(":", 1) for p in args.actor_remap.split(",")) if args.actor_remap else None
    r = run(
        args.task,
        args.shipped,
        args.demo,
        out,
        goal_actor=args.goal_actor,
        max_scan=args.max_scan,
        episode=args.episode,
        render_attempts=args.render_attempts,
        actor_remap=remap,
        rank_by=args.rank_by,
        truncate_buffer=None if args.truncate_buffer < 0 else args.truncate_buffer,
    )
    print(
        f"{r['task']} [{r['episode']}]: success@{r['success_step']} steps={r['steps']} "
        f"mean_pixel_diff={r['mean_pixel_diff']:.4f}/255 -> {r['out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
