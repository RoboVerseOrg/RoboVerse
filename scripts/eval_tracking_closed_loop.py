"""Closed-loop eval of mjlab's pretrained G1 *tracking* (LAFAN dance) policy in RoboVerse.

Loads the canonical mjlab rsl_rl tracking checkpoint (``demo_ckpt.pt``: an
``ActorCritic`` whose actor is EmpiricalNormalization -> MLP(512,256,128,ELU),
160-D obs, 29-D action), reconstructs the deterministic-mean actor standalone,
and runs it closed-loop in the RoboVerse ``mjlab.tracking_flat_g1_v2`` env with
the LAFAN dance motion.

Parity with mjlab "play" mode:
  * the robot is reset-state-injected (RSI) to the motion's frame-0 reference
    pose (root pose + joint pos/vel), exactly like mjlab ``_resample_command``
    with ``sampling_mode="start"``;
  * the MotionCommandManager's ``time_steps`` starts at 0 and advances by 1 per
    control step (already done in ``MotionCommandManager.update``);
  * domain-randomization push events are disabled.

Reports base-height + per-step anchor / body tracking error vs the motion
reference, and (with ``--ref``) diffs against an mjlab-native rollout npz
produced by ``scripts/_track_native_rollout.py``.

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES=1 \
    MJLAB_G1_MOTION_FILE=/tmp/claude-0/mjlab_cache/lafan1_dance1_subject1_demo_motion.npz \
    python scripts/eval_tracking_closed_loop.py \
        --ckpt /tmp/claude-0/mjlab_cache/demo_ckpt.pt --steps 400 \
        --ref /tmp/claude-0/track_native.npz \
        --render tools/mjlab_integration/policy_replay/closed_loop/g1_dance_roboverse.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")

OBS_DIM = 160
ACT_DIM = 29
HIDDEN = (512, 256, 128)
TASK = "mjlab.tracking_flat_g1_v2"


class StandaloneTrackingPolicy(nn.Module):
    """mjlab rsl_rl tracking actor: EmpiricalNormalization -> MLP(ELU), det. mean.

    Handles the canonical (legacy) checkpoint layout used by ``demo_ckpt.pt``:
    ``model_state_dict`` with ``actor.{0,2,4,6}.*`` linear layers and
    ``actor_obs_normalizer.{_mean,_std}`` — the same migration mjlab's
    ``runner.load`` applies (actor.* -> mlp.*, actor_obs_normalizer.* ->
    obs_normalizer.*).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden, device) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, obs_dim))
        self.register_buffer("_std", torch.ones(1, obs_dim))
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden[0]), nn.ELU()]
        for i in range(len(hidden) - 1):
            layers += [nn.Linear(hidden[i], hidden[i + 1]), nn.ELU()]
        layers.append(nn.Linear(hidden[-1], act_dim))
        self.mlp = nn.Sequential(*layers)
        self.eps = 1e-8
        self.to(device)
        self.device = device

    def load_from_ckpt(self, ckpt_path: str) -> int:
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = state.get("model_state_dict", state.get("actor_state_dict"))
        # Normalizer (actor side).
        for src in ("actor_obs_normalizer", "obs_normalizer"):
            if f"{src}._mean" in sd:
                self._mean[:] = sd[f"{src}._mean"].to(self.device).reshape(1, -1)
                self._std[:] = sd[f"{src}._std"].to(self.device).reshape(1, -1)
                break
        # Actor MLP: keys like ``actor.0.weight`` (or ``mlp.0.weight``).
        prefix = "actor." if any(k.startswith("actor.") for k in sd) else "mlp."
        mlp_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = self.mlp.load_state_dict(mlp_sd, strict=False)
        assert not [m for m in missing if "weight" in m or "bias" in m], f"missing {missing}"
        self.eval()
        return int(state.get("iter", -1))

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.to(self.device)
        x = (x - self._mean) / (self._std + self.eps)
        return self.mlp(x)


def _rsi_to_motion_frame0(env, frame: int = 0) -> None:
    """Inject the motion's reference state at ``frame`` into the sim (mjlab RSI).

    Mirrors mjlab ``MotionCommand._write_reference_state_to_sim`` for
    ``sampling_mode="start"``: write root pose (pelvis = first tracked body),
    root + joint velocities, and joint positions from the motion clip, then set
    the command's ``time_steps`` to ``frame``.
    """
    import mujoco

    mc = env.command_managers["motion"]
    if mc.motion._is_identity:
        return
    ph = env.handler.physics
    mp = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
    data = ph.data._data if hasattr(ph.data, "_data") else ph.data

    # Motion frame-0 root (pelvis is body_names[0]) and joint reference.
    root_pos = mc.motion.body_pos_w[frame, 0].detach().cpu().numpy()
    root_quat = mc.motion.body_quat_w[frame, 0].detach().cpu().numpy()  # wxyz
    root_lin = mc.motion.body_lin_vel_w[frame, 0].detach().cpu().numpy()
    root_ang = mc.motion.body_ang_vel_w[frame, 0].detach().cpu().numpy()
    jpos = mc.motion.joint_pos[frame].detach().cpu().numpy()
    jvel = mc.motion.joint_vel[frame].detach().cpu().numpy()

    qpos = np.asarray(data.qpos).copy()
    qvel = np.asarray(data.qvel).copy()
    qpos[0:3] = root_pos
    qpos[3:7] = root_quat
    qpos[7:7 + ACT_DIM] = jpos
    qvel[0:3] = root_lin
    qvel[3:6] = root_ang
    qvel[6:6 + ACT_DIM] = jvel
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(mp, data)
    mc.time_steps[:] = frame


def _make_camera(model, body: str):
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    cam.trackbodyid = bid
    cam.distance = 3.0
    cam.elevation = -8.0
    cam.azimuth = 120.0
    return cam


def run(ckpt: str, steps: int, render: str | None, no_term: bool):
    import roboverse_pack  # noqa: F401  register tasks
    from metasim.task.registry import get_task_class

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES pins the physical GPU
    torch.manual_seed(0)
    np.random.seed(0)

    cls = get_task_class(TASK)
    env = cls(device=device)

    # Play parity: drop push DR.
    if getattr(env, "post_step_events", None):
        env.post_step_events = {k: v for k, v in env.post_step_events.items() if "push" not in k.lower()}
    # Play parity: mjlab sets episode_length_s ~ 1e9 in play so the time_out
    # termination never fires mid-clip (env_cfgs.py:89). Without this the env
    # resets at 20s (1000 steps) and the MotionCommandManager re-samples a random
    # motion frame, teleporting the reference and desyncing the dance. Keep the
    # derived ``max_episode_steps`` within int32 range (the time_out term compares
    # an int32 step counter against it): 1e6 s = 5e7 steps is huge yet safe.
    env.cfg.max_episode_length_s = 1.0e6
    # Optionally disable terminations so a transient tracking slip doesn't reset
    # the motion phase (mjlab play keeps terminations but the policy rarely
    # trips them; --no-term isolates pure tracking behaviour).
    if no_term:
        env.cfg.terminations = type(env.cfg.terminations)()

    policy = StandaloneTrackingPolicy(OBS_DIM, ACT_DIM, HIDDEN, device)
    it = policy.load_from_ckpt(ckpt)

    env.reset()
    mc = env.command_managers["motion"]
    assert not mc.motion._is_identity, "motion file not loaded (set MJLAB_G1_MOTION_FILE)"
    print(f"[roboverse] ckpt={ckpt} iter={it} motion_T={mc.motion.time_step_total}")

    # RSI: start at motion frame 0 and recompute obs from that state.
    _rsi_to_motion_frame0(env, frame=0)
    obs = _actor(env._observation(env.handler.get_states(mode="tensor")))

    ph = env.handler.physics
    z_log = [float(ph.data.qpos[2])]
    anchor_err, body_err, robot_jpos, ref_jpos, time_idx = [], [], [], [], []

    renderer = cam = mj_data = None
    frames: list = []
    if render is not None:
        import mujoco

        model = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = _make_camera(model, "torso_link")
        mj_data = ph.data._data if hasattr(ph.data, "_data") else ph.data

    for _ in range(steps):
        # Snapshot tracking error against the CURRENT reference frame, before step.
        a_err = float(torch.norm(mc.anchor_pos_w - mc.robot_anchor_pos_w, dim=-1).mean())
        b_err = float(torch.norm(mc.body_pos_relative_w - mc.robot_body_pos_w, dim=-1).mean())
        anchor_err.append(a_err)
        body_err.append(b_err)
        robot_jpos.append(_robot_jpos(env))
        ref_jpos.append(mc.joint_pos[0].detach().cpu().numpy().copy())
        time_idx.append(int(mc.time_steps[0].item()))

        with torch.inference_mode():
            action = policy(obs)
        out = env.step(action)
        obs = _actor(out[0])
        z_log.append(float(ph.data.qpos[2]))
        if renderer is not None:
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())
        if float(ph.data.qpos[2]) < 0.2:
            print(f"[roboverse] collapsed at step {len(z_log)} z={z_log[-1]:.3f}")
            break

    z = np.array(z_log)
    anchor_err = np.array(anchor_err)
    body_err = np.array(body_err)
    robot_jpos = np.array(robot_jpos)
    ref_jpos = np.array(ref_jpos)
    time_idx = np.array(time_idx)

    if render is not None and frames:
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(render)), exist_ok=True)
        imageio.mimwrite(render, frames, fps=50, codec="libx264", quality=7)
        print(f"[roboverse] wrote {render} ({len(frames)} frames @ 50 fps)")

    env.close()
    return dict(z=z, anchor_err=anchor_err, body_err=body_err,
                robot_jpos=robot_jpos, ref_jpos=ref_jpos, time_idx=time_idx)


def _actor(obs):
    """Extract the actor obs tensor (the env may return a dict of obs groups)."""
    if isinstance(obs, dict):
        return obs["actor"]
    return obs


def _robot_jpos(env) -> np.ndarray:
    """Robot joint positions in mjlab joint order (qpos[7:36])."""
    ph = env.handler.physics
    return np.asarray(ph.data.qpos[7:7 + ACT_DIM], dtype=np.float32).copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/tmp/claude-0/mjlab_cache/demo_ckpt.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--render", default=None)
    p.add_argument("--ref", default=None, help="mjlab-native rollout npz for comparison")
    p.add_argument("--no-term", action="store_true", help="disable terminations")
    p.add_argument("--out", default=None, help="save RoboVerse rollout npz")
    args = p.parse_args()

    r = run(args.ckpt, args.steps, args.render, args.no_term)
    z = r["z"]
    up = bool(z.min() > 0.5 and z[-1] > 0.5)
    print(f"\n[RoboVerse tracking] steps={len(z)-1}")
    print(f"  base z: z0={z[0]:.3f} min={z.min():.3f} mean={z.mean():.3f} final={z[-1]:.3f} -> {'UP' if up else 'COLLAPSED'}")
    print(f"  anchor_err: mean={r['anchor_err'].mean():.4f} max={r['anchor_err'].max():.4f}")
    print(f"  body_err:   mean={r['body_err'].mean():.4f} max={r['body_err'].max():.4f}")

    if args.out:
        np.savez(args.out, **r)
        print(f"  wrote {args.out}")

    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref)
        n = min(len(z), len(ref["z"]))
        dz = np.abs(z[:n] - ref["z"][:n])
        nj = min(len(r["robot_jpos"]), len(ref["robot_jpos"]))
        djp = np.abs(r["robot_jpos"][:nj] - ref["robot_jpos"][:nj])
        print("\n[RoboVerse vs mjlab-native]")
        print(f"  steps native={len(ref['z'])-1} roboverse={len(z)-1}")
        print(f"  base z      native: min={ref['z'].min():.3f} mean={ref['z'].mean():.3f} final={ref['z'][-1]:.3f}")
        print(f"  |dz|        mean={dz.mean():.4f} max={dz.max():.4f}")
        print(f"  anchor_err  native mean={ref['anchor_err'].mean():.4f}  roboverse mean={r['anchor_err'].mean():.4f}")
        print(f"  body_err    native mean={ref['body_err'].mean():.4f}  roboverse mean={r['body_err'].mean():.4f}")
        print(f"  |d joint_pos| (robot, per-step mean over joints) mean={djp.mean():.4f} max={djp.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
