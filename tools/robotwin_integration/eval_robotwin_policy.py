"""Closed-loop policy evaluation in native RoboTwin (the policy-reproduction harness).

Runs in the ``robotwin`` conda env. The native passthrough env is the *only*
closed-loop RoboTwin environment (RoboVerse has no RoboTwin handler env), so this
is where a learned policy must be evaluated to claim it reproduces RoboTwin's
results. A policy maps RoboTwin's per-step observation (head-camera RGB + joint
state) to a 14-D joint action ``[L_arm(6), L_grip, R_arm(6), R_grip]``, which is
rolled out through ``env.take_action(action, 'qpos')`` -- the *exact* closed-loop
interface RoboTwin's own ``script/eval_policy.py`` uses (TOPP-interpolates the
waypoint, steps physics, sets ``eval_success`` when ``check_success()`` fires).
Success rate over N seeds is directly comparable to RoboTwin's expert.

Policies (``--policy``):

- ``replay`` -- open-loop action replay: emit the bridge-recorded command stream
  ``vectors[t]`` one waypoint per ``take_action``. Needs no model/GPU and answers
  a sharp 1:1 question: does RoboTwin's recorded bimanual action trajectory, driven
  through ``take_action``'s TOPP (not the original curobo plan that produced it),
  still complete the task? It is also the harness self-test the DP path reuses.
- ``dp`` -- a RoboVerse-trained Diffusion Policy / ACT checkpoint
  (``roboverse_learn/il``). Loaded in-process if importable in this env, else point
  ``--server`` at a ``policy_model_server`` running in the ``roboverse`` env.

Run (robotwin env)::

    # open-loop action-replay baseline (the recorded trajectory, via take_action)
    conda run -n robotwin env MUJOCO_GL=egl SAPIEN_HEADLESS=1 python \\
        tools/robotwin_integration/eval_robotwin_policy.py --task beat_block_hammer \\
        --policy replay --bridge ~/projects/robotwin/data/_rv_bridge/beat_block_hammer.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PASSTHROUGH = os.path.join(_HERE, "..", "..", "roboverse_pack", "tasks", "robotwin", "_passthrough.py")


def _load_passthrough():
    spec = importlib.util.spec_from_file_location("rt_passthrough", _PASSTHROUGH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Policy:
    """Maps a RoboTwin observation to a 14-D joint action (or None to stop)."""

    def reset(self) -> None: ...

    def predict(self, obs: dict) -> np.ndarray | None:
        raise NotImplementedError

    def needs_obs(self) -> bool:
        """Whether the NEXT predict() call will actually consume the observation.

        Lets the eval loop skip the (expensive, RT-rendered) ``get_obs()`` on steps where
        the policy will just replay a buffered action. Default True = render every step
        (correct for any policy). Chunked policies override this to render only at chunk
        boundaries -- a pure efficiency win that does not change the obs the policy sees.
        """
        return True


class ReplayPolicy(Policy):
    """Emit the bridge-recorded command targets one per step (open-loop replay)."""

    def __init__(self, bridge: dict):
        self.vectors = [np.asarray(v, dtype=float) for v in bridge["vectors"]]
        self.seed = int(bridge.get("seed", 0))
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def predict(self, obs: dict) -> np.ndarray | None:
        if self._t >= len(self.vectors):
            return None  # trajectory exhausted -> let the episode end
        a = self.vectors[self._t]
        self._t += 1
        return a


class DPPolicy(Policy):
    """A RoboVerse-trained DP checkpoint, served from the roboverse env.

    The DP model + its deps run in the ``roboverse`` env (``dp_policy_server.py``);
    this client (``robotwin`` env) sends RoboTwin's head-camera RGB + joint state and
    receives a 14-D joint action chunk, caching it and emitting one action per step
    (chunked inference, matching DefaultEvalRunner.get_action). Env-decoupled the way
    RoboTwin's own policy server/client split is, so the conflicting SAPIEN/torch
    builds never share a process.
    """

    def __init__(self, server: str, camera: str = "head_camera", timeout: float = 120.0):
        import socket as _socket

        host, _, port = server.partition(":")
        self._addr = (host or "127.0.0.1", int(port or "5599"))
        self.camera = camera
        self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        self._sock.connect(self._addr)
        # A single DP inference is ~seconds; a >2min wait means the server hung
        # (a CUDA/render stall we saw deadlock a whole eval). Time out so the episode
        # fails gracefully instead of blocking the sweep forever.
        self._sock.settimeout(timeout)
        self._cache: list = []
        info = self._rpc({"cmd": "ping"})
        self.n_action_steps = int(info.get("n_action_steps", 1))
        # The server reports which RoboTwin cameras its (possibly multi-view) ckpt needs;
        # fall back to the single primary camera for an old single-view server.
        self.cameras = list(info.get("cameras") or [camera])
        print(f"[dp client] connected to {self._addr}, n_action_steps={self.n_action_steps}, cameras={self.cameras}")

    # --- length-prefixed pickle RPC (mirrors dp_policy_server) ---
    def _rpc(self, obj):
        import struct

        body = pickle.dumps(obj)
        self._sock.sendall(struct.pack(">I", len(body)) + body)
        header = self._recv_exactly(4)
        (length,) = struct.unpack(">I", header)
        return pickle.loads(self._recv_exactly(length))

    def _recv_exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("dp_policy_server closed the connection")
            buf += chunk
        return buf

    def reset(self) -> None:
        self._cache = []
        self._rpc({"cmd": "reset"})

    def needs_obs(self) -> bool:
        # Only when the action buffer is empty do we query the server (and thus read the
        # head-camera obs); on the other n_action_steps-1 steps we replay a cached action
        # and ignore obs -- so the eval can skip rendering it entirely.
        return not self._cache

    def predict(self, obs: dict) -> np.ndarray | None:
        if not self._cache:
            try:
                cams = {c: np.asarray(obs["observation"][c]["rgb"]) for c in self.cameras}
            except (KeyError, TypeError) as e:
                raise RuntimeError(
                    f"obs missing observation[{self.cameras}]['rgb'] -- run the env with data_type rgb=True "
                    "and ensure the wrist cameras are enabled for a multi-view ckpt"
                ) from e
            jq = self._robotwin_joint_qpos(obs)
            resp = self._rpc({"cmd": "predict", "cameras": cams, "joint_qpos": jq})
            if "error" in resp:
                raise RuntimeError(f"dp server predict failed: {resp['error']}\n{resp.get('traceback', '')}")
            self._cache = [np.asarray(a, dtype=float) for a in resp["action_chunk"]]
        return self._cache.pop(0) if self._cache else None

    @staticmethod
    def _robotwin_joint_qpos(obs: dict) -> np.ndarray:
        """Pull the 14-D [L_arm(6), L_grip, R_arm(6), R_grip] state from RoboTwin obs.

        Prefer the achieved state injected by the eval loop (``robot_joint_state``,
        read from the robot exactly like collect_bridge's training ``real_vector``);
        fall back to a script-populated ``joint_action`` if present.
        """
        if obs.get("robot_joint_state") is not None:
            return np.asarray(obs["robot_joint_state"], dtype=float)
        ja = obs.get("joint_action", {})
        v = ja.get("real_vector", ja.get("vector"))
        return np.asarray(v, dtype=float)


def _build_policy(args) -> Policy:
    if args.policy == "replay":
        if not args.bridge:
            raise SystemExit("--policy replay requires --bridge <pkl>")
        with open(os.path.expanduser(args.bridge), "rb") as f:
            return ReplayPolicy(pickle.load(f))
    if args.policy == "dp":
        if not args.server:
            raise SystemExit(
                "--policy dp requires --server <host:port> (start dp_policy_server.py in the roboverse env)"
            )
        return DPPolicy(args.server, args.camera)
    raise SystemExit(f"unknown policy {args.policy!r}")


def _robotwin_step_lim(pt, task: str) -> int:
    """RoboTwin's per-task eval step budget (task_config/_eval_step_limit.yml).

    Matching this is part of evaluating "the same as RoboTwin": its own eval judges
    a policy over exactly this many steps (e.g. beat_block_hammer = 400). The env
    only loads it when eval_mode=True (which also switches to unseen textures); we
    want the real budget while keeping training-consistent (seen) textures, so we
    read the table directly. Falls back to 1000 if the task isn't listed.
    """
    try:
        import yaml

        path = os.path.join(pt.robotwin_dir(), "task_config", "_eval_step_limit.yml")
        with open(path) as f:
            return int(yaml.safe_load(f).get(task, 1000))
    except Exception:
        return 1000


def _eval_one(
    pt, task: str, task_config: str, seed: int, policy: Policy, rgb: bool, step_lim: int, video_path: str | None = None
) -> tuple[bool, int]:
    """Run one closed-loop episode; return (success, steps_taken).

    If ``video_path`` is set, capture a world-up observer camera every action step and
    write an mp4 -- a diagnostic to SEE what the policy does (e.g. why a task fails).
    """
    data_type = {k: False for k in ("rgb", "third_view", "depth", "pointcloud", "observer", "endpose", "qpos")}
    data_type["rgb"] = rgb  # the policy may consume head_camera RGB
    env = pt._make_robotwin_env(
        task_name=task, task_config=task_config, seed=seed, is_test=True, data_type=data_type, render_freq=0,
    )  # fmt: skip
    _vframes: list = []
    _vcam = None
    if video_path:
        try:
            import sapien

            pos = np.array([0.0, 0.6, 1.35]); look = np.array([0.0, -0.3, 0.78])
            fwd = look - pos; fwd /= np.linalg.norm(fwd) + 1e-9
            left = np.cross([0.0, 0.0, 1.0], fwd); left /= np.linalg.norm(left) + 1e-9
            up = np.cross(fwd, left)
            mat = np.eye(4); mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = fwd, left, up, pos
            _vcam = env.scene.add_camera(name="diag", width=512, height=512, fovy=np.deg2rad(55), near=0.05, far=100)
            _vcam.entity.set_pose(sapien.Pose(mat))
        except Exception as e:
            print(f"[video] camera setup failed: {type(e).__name__}: {e}")
            _vcam = None
    if rgb:
        # setup_demo enables the RT shader + OIDN denoiser; the DP eval renders the
        # head_camera every step, and OIDN deadlocks headless on a long episode
        # ("OIDN Error: invalid handle") -- that's what hung whole evals. Drop the
        # denoiser (RT geometry still renders) -- same fix as native_render/collect,
        # and it matches the demos (collected with OIDN off) for train/eval parity.
        try:
            import sapien

            sapien.render.set_ray_tracing_denoiser("none")
        except Exception:
            pass
    try:
        policy.reset()
    except (TimeoutError, OSError) as e:
        print(f"[{task} seed {seed}] policy.reset timeout/IO ({type(e).__name__}); marking FAIL")
        env.close_env()
        return False, 0
    try:
        # Use RoboTwin's real per-task budget (env.step_lim is only set under
        # eval_mode, which we avoid to keep seen textures); fall back to the table.
        step_lim = int(env.step_lim) if getattr(env, "step_lim", None) else step_lim
        while env.take_action_cnt < step_lim:
            # Render the observation only when the policy will actually use it. A chunked
            # DP consumes obs once per action chunk and replays buffered actions in between;
            # since get_obs() renders the head-camera with the (intermittently deadlocking)
            # RT shader, skipping it on the in-between steps cuts RT renders ~n_action_steps×
            # -- far fewer chances to hit the upstream hang, and a big speedup -- WITHOUT
            # changing the obs the policy sees at chunk boundaries.
            obs = None
            if getattr(policy, "needs_obs", lambda: True)():
                obs = env.get_obs()
                # Inject the achieved 14-D joint state [L_arm(6), L_grip, R_arm(6), R_grip]
                # the SAME way collect_bridge records `real_vector` (the DP's training
                # state obs): in eval mode obs["joint_action"] isn't script-populated, so
                # read it straight from the robot for train/eval consistency.
                try:
                    obs["robot_joint_state"] = np.asarray(
                        list(env.robot.get_left_arm_real_jointState())
                        + list(env.robot.get_right_arm_real_jointState()),
                        dtype=float,
                    )
                except Exception:
                    pass
            try:
                action = policy.predict(obs)
            except (TimeoutError, OSError) as e:
                # Server stalled (CUDA/render hang) -> abandon this episode as a fail
                # rather than deadlock the whole sweep. socket.timeout is an OSError.
                print(f"[{task} seed {seed}] policy timeout/IO ({type(e).__name__}: {e}); marking FAIL")
                break
            if action is None:
                break
            env.take_action(np.asarray(action, dtype=float), action_type="qpos")
            if _vcam is not None:
                try:
                    env.scene.update_render()
                    _vcam.take_picture()
                    px = np.asarray(_vcam.get_picture("Color"))[..., :3]
                    _vframes.append((np.clip(px, 0, 1) * 255).astype(np.uint8))
                except Exception:
                    pass
            if env.eval_success:
                break
        succ = bool(env.eval_success)
        steps = int(env.take_action_cnt)
    finally:
        env.close_env()
    if video_path and _vframes:
        try:
            import imageio.v2 as imageio

            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            imageio.mimsave(video_path, _vframes, fps=20)
            print(f"[video] saved {len(_vframes)} frames -> {video_path} (success={succ})")
        except Exception as e:
            print(f"[video] save failed: {type(e).__name__}: {e}")
    return succ, steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="RoboTwin task name (envs/<task>.py)")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--policy", choices=["replay", "dp"], default="replay")
    ap.add_argument("--bridge", default=None, help="bridge pkl (for --policy replay; uses its recorded seed)")
    ap.add_argument("--server", default=None, help="dp_policy_server host:port (for --policy dp; default port 5599)")
    ap.add_argument("--camera", default="head_camera", help="image obs camera for --policy dp")
    ap.add_argument("--num-eval", type=int, default=1, help="episodes to evaluate")
    ap.add_argument("--start-seed", type=int, default=None, help="first seed (default: bridge seed for replay, else 0)")
    ap.add_argument("--video", default=None, help="diagnostic: save an observer mp4 of the FIRST episode to this path")
    args = ap.parse_args(argv)

    pt = _load_passthrough()
    policy = _build_policy(args)
    needs_rgb = args.policy == "dp"

    # Replay is only meaningful at the seed the trajectory was recorded for (object
    # placement is seed-dependent); default to the bridge's seed.
    if args.start_seed is not None:
        base_seed = args.start_seed
    elif isinstance(policy, ReplayPolicy):
        base_seed = policy.seed
    else:
        base_seed = 0

    step_lim = _robotwin_step_lim(pt, args.task)
    print(f"[{args.task}] eval step budget = {step_lim} (RoboTwin's per-task limit)")

    successes = 0
    for i in range(args.num_eval):
        seed = base_seed + i
        vpath = args.video if (args.video and i == 0) else None
        succ, steps = _eval_one(pt, args.task, args.task_config, seed, policy, needs_rgb, step_lim, video_path=vpath)
        successes += int(succ)
        print(f"[{args.task} seed {seed}] {'SUCCESS' if succ else 'FAIL'} ({steps} steps)")

    rate = successes / max(1, args.num_eval)
    print(f"RESULT {args.task} | policy={args.policy} | success {successes}/{args.num_eval} = {rate * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
