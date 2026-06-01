"""Closed-loop eval of an mjlab-trained velocity policy in the RoboVerse env.

Loads an mjlab rsl_rl checkpoint (ActorCritic, with the trained obs normalizer),
runs it closed-loop in the RoboVerse g1/go1 velocity env under a FIXED forward
command for N control steps, and reports whether the robot stays up (base height
~0.7+ for g1 / ~0.27+ for go1, not <0.3 / <0.12) plus mean reward.

It also rolls the SAME policy out in mjlab-native (via the mjlab ManagerBasedRlEnv)
from a matched init + the same fixed command, and prints the base-height and
base-forward-velocity trajectories side by side so divergence is visible with
numbers (true closed-loop 1:1 check, not obs-bitwise).

Usage:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_closed_loop.py \
        --robot g1 --steps 300 --cmd-vx 1.0

The standalone policy (mlp + EmpiricalNormalization) is reconstructed from the
checkpoint's ``actor_state_dict`` so no mjlab env build is needed for the
RoboVerse rollout; the mjlab-native rollout (``--with-mjlab``) builds the mjlab
env and uses its own runner for an apples-to-apples reference.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")

_ROBOTS = {
    "g1": {
        "task": "mjlab.velocity_flat_g1_v2",
        "mjlab_task": "Mjlab-Velocity-Flat-Unitree-G1",
        "obs_dim": 99,
        "act_dim": 29,
        "hidden": (512, 256, 128),
        "obs_norm": True,
        "ckpt_glob": "/workspace/mjlab_upstream/logs/rsl_rl/g1_velocity/*/model_*.pt",
        "fall_z": 0.3,
        "stand_z": 0.6,
    },
    "go1": {
        "task": "mjlab.velocity_flat_go1_v2",
        "mjlab_task": "Mjlab-Velocity-Flat-Unitree-Go1",
        "obs_dim": 48,
        "act_dim": 12,
        "hidden": (512, 256, 128),
        "obs_norm": False,
        "ckpt_glob": "/workspace/mjlab_upstream/logs/rsl_rl/go1_velocity/*/model_*.pt",
        "fall_z": 0.12,
        "stand_z": 0.22,
    },
}


def _latest_ckpt(pattern: str) -> str:
    """Highest-iteration ``model_*.pt`` matching the glob."""
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"no checkpoint matched {pattern}")

    def it(p: str) -> int:
        base = os.path.basename(p)
        try:
            return int(base.replace("model_", "").replace(".pt", ""))
        except ValueError:
            return -1

    return max(paths, key=it)


class StandalonePolicy(nn.Module):
    """mjlab rsl_rl actor: EmpiricalNormalization -> MLP (ELU). Deterministic mean."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...], obs_norm: bool, device) -> None:
        super().__init__()
        self.obs_norm = obs_norm
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

    def load_from_ckpt(self, ckpt_path: str) -> float:
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        asd = state["actor_state_dict"]
        if self.obs_norm:
            self._mean[:] = asd["obs_normalizer._mean"].to(self.device).reshape(1, -1)
            self._std[:] = asd["obs_normalizer._std"].to(self.device).reshape(1, -1)
        mlp_sd = {k[len("mlp.") :]: v for k, v in asd.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_sd)
        self.eval()
        return int(state.get("iter", -1))

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.to(self.device)
        if self.obs_norm:
            x = (x - self._mean) / (self._std + self.eps)
        return self.mlp(x)


def _force_command(cmd_mgr, vx: float, vy: float, wz: float, device) -> None:
    """Pin a velocity-command manager to a constant twist (disable resample/heading)."""
    n = cmd_mgr._command.shape[0]
    cmd_mgr._command[:] = torch.tensor([vx, vy, wz], device=device).reshape(1, 3).repeat(n, 1)
    if hasattr(cmd_mgr, "_next_resample_time"):
        cmd_mgr._next_resample_time[:] = float("inf")
    if hasattr(cmd_mgr, "_is_heading"):
        cmd_mgr._is_heading[:] = False
    if hasattr(cmd_mgr, "_is_standing"):
        cmd_mgr._is_standing[:] = False


_TRACK_BODY = {"g1": "pelvis", "go1": "trunk"}


def _make_tracking_camera(model, body: str):
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    cam.trackbodyid = bid
    cam.fixedcamid = -1
    cam.distance = 2.6
    cam.elevation = -12.0
    cam.azimuth = 90.0
    return cam


def run_roboverse(
    robot: str,
    ckpt: str,
    steps: int,
    cmd: tuple[float, float, float],
    seed: int,
    render_path: str | None = None,
    width: int = 640,
    height: int = 480,
):
    import roboverse_pack  # noqa: F401  register tasks
    from metasim.task.registry import get_task_class

    spec = _ROBOTS[robot]
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cls = get_task_class(spec["task"])
    env = cls(device=device)
    policy = StandalonePolicy(spec["obs_dim"], spec["act_dim"], spec["hidden"], spec["obs_norm"], device)
    it = policy.load_from_ckpt(ckpt)
    print(f"[roboverse:{robot}] checkpoint={ckpt} (iter={it})")

    # Eval = mjlab "play" mode: disable domain randomization so the closed-loop
    # rollout is deterministic and comparable to mjlab-native play (no random
    # per-step velocity pushes, no DR friction/COM/encoder-bias jitter).
    if getattr(env, "post_step_events", None):
        env.post_step_events = {k: v for k, v in env.post_step_events.items() if "push" not in k.lower()}

    env.reset()
    cmd_mgr = env.command_managers["twist"]
    _force_command(cmd_mgr, *cmd, device)

    ph = env.handler.physics
    z_log, vx_log, rew_log = [float(ph.data.qpos[2])], [float(ph.data.qvel[0])], []
    obs = env._obs_buf["actor"]

    renderer = cam = mj_data = None
    frames: list = []
    if render_path is not None:
        import mujoco

        model = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
        renderer = mujoco.Renderer(model, width=width, height=height)
        cam = _make_tracking_camera(model, _TRACK_BODY[robot])
        mj_data = ph.data._data if hasattr(ph.data, "_data") else ph.data

    for _ in range(steps):
        _force_command(cmd_mgr, *cmd, device)  # re-pin (update() may have touched it)
        with torch.inference_mode():
            action = policy(obs)
        out = env.step(action)
        _force_command(cmd_mgr, *cmd, device)
        obs = out[0]["actor"]
        z_log.append(float(ph.data.qpos[2]))
        vx_log.append(float(ph.data.qvel[0]))
        rew_log.append(float(out[1].mean()))
        if renderer is not None:
            renderer.update_scene(mj_data, camera=cam if cam is not None else -1)
            frames.append(renderer.render().copy())
        if float(ph.data.qpos[2]) < 0.05:  # fully collapsed
            break

    if render_path is not None and frames:
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(render_path)), exist_ok=True)
        imageio.mimwrite(render_path, frames, fps=50, codec="libx264", quality=7)
        print(f"[roboverse:{robot}] wrote {render_path} ({len(frames)} frames @ 50 fps)")
    env.close()
    return np.array(z_log), np.array(vx_log), np.array(rew_log)


def run_mjlab(robot: str, ckpt: str, steps: int, cmd: tuple[float, float, float]):
    """mjlab-native closed-loop rollout under the same fixed command."""
    sys.path.insert(0, os.path.join(os.environ.get("MJLAB_REPO", "/workspace/mjlab_upstream"), "src"))
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    spec = _ROBOTS[robot]
    device = "cuda:0"
    env_cfg = load_env_cfg(spec["mjlab_task"], play=True)
    agent_cfg = load_rl_cfg(spec["mjlab_task"])
    env_cfg.scene.num_envs = 1
    base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(spec["mjlab_task"]) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd_mgr = base.command_manager.get_term("twist") if hasattr(base, "command_manager") else None

    cmd_t = torch.tensor(cmd, device=device).reshape(1, 3)

    def pin():
        # mjlab UniformVelocityCommand stores the policy-visible command in
        # ``vel_command_b`` (``.command`` is a read-only property onto it); pin
        # that + the world copy, disable heading/standing, and push the resample
        # timer out so the constant forward command holds for the whole rollout.
        if cmd_mgr is None:
            return
        for attr in ("vel_command_b", "vel_command_w"):
            if hasattr(cmd_mgr, attr):
                getattr(cmd_mgr, attr)[:] = cmd_t
        for attr in ("is_standing_env", "is_heading_env"):
            if hasattr(cmd_mgr, attr):
                getattr(cmd_mgr, attr)[:] = False
        if hasattr(cmd_mgr, "time_left"):
            cmd_mgr.time_left[:] = 1e9

    sd = base.sim.data
    robot = base.scene["robot"]

    def _body_vx() -> float:
        # BODY-frame forward velocity (matches the RoboVerse side, which logs
        # root_link_lin_vel_b). Logging world-frame qvel[0] here would be a
        # frame mismatch — if the gait drifts in heading, world-x << body-x.
        return float(robot.data.root_link_lin_vel_b[0, 0].item())

    z_log, vx_log = [float(sd.qpos[0, 2].item())], [_body_vx()]
    obs = env.get_observations()
    pin()
    for _ in range(steps):
        pin()
        with torch.inference_mode():
            actions = policy(obs)
        obs = env.step(actions)[0]
        pin()
        z_log.append(float(sd.qpos[0, 2].item()))
        vx_log.append(_body_vx())
    env.close()
    return np.array(z_log), np.array(vx_log)


def _summary(tag: str, z: np.ndarray, vx: np.ndarray, fall_z: float, stand_z: float, rew=None) -> None:
    up = bool(z.min() > fall_z and z[-1] > fall_z)
    status = "UP" if up else "COLLAPSED"
    print(f"\n[{tag}] steps={len(z) - 1}")
    print(f"  base height: z0={z[0]:.3f} min={z.min():.3f} mean={z.mean():.3f} final={z[-1]:.3f} -> {status}")
    print(f"  base vx:     mean={vx.mean():.3f} final={vx[-1]:.3f} (fall_z={fall_z}, stand_z={stand_z})")
    if rew is not None and len(rew):
        print(f"  reward:      mean/step={rew.mean():.3f}  episode_sum={rew.sum():.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=("g1", "go1"), required=True)
    p.add_argument("--ckpt", default=None, help="checkpoint .pt (default: highest-iter for robot)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--cmd-vx", type=float, default=1.0)
    p.add_argument("--cmd-vy", type=float, default=0.0)
    p.add_argument("--cmd-wz", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-mjlab", action="store_true", help="also run mjlab-native reference rollout")
    p.add_argument("--render", default=None, help="write a RoboVerse policy-driven mp4 to this path")
    args = p.parse_args()

    spec = _ROBOTS[args.robot]
    ckpt = args.ckpt or _latest_ckpt(spec["ckpt_glob"])
    cmd = (args.cmd_vx, args.cmd_vy, args.cmd_wz)
    print(f"=== closed-loop eval: robot={args.robot} cmd(vx,vy,wz)={cmd} steps={args.steps} ===")

    rz, rvx, rrew = run_roboverse(args.robot, ckpt, args.steps, cmd, args.seed, render_path=args.render)
    _summary("RoboVerse", rz, rvx, spec["fall_z"], spec["stand_z"], rrew)

    if args.with_mjlab:
        mz, mvx = run_mjlab(args.robot, ckpt, args.steps, cmd)
        _summary("mjlab-native", mz, mvx, spec["fall_z"], spec["stand_z"])
        nmin = min(len(rz), len(mz))
        dz = np.abs(rz[:nmin] - mz[:nmin])
        dvx = np.abs(rvx[:nmin] - mvx[:nmin])
        print("\n[tracking RoboVerse vs mjlab over common steps]")
        print(f"  |dz|   mean={dz.mean():.4f} max={dz.max():.4f}")
        print(f"  |dvx|  mean={dvx.mean():.4f} max={dvx.max():.4f}")
        print(f"  vx target={cmd[0]:.2f}  RoboVerse mean_vx={rvx.mean():.3f}  mjlab mean_vx={mvx.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
