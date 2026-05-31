"""Serve a RoboVerse-trained Diffusion Policy for closed-loop RoboTwin eval.

The trained DP checkpoint lives in RoboVerse's IL stack and its inference deps
(diffusers/einops/dill + the numba-optional dataset) run in the ``roboverse`` env,
but the *only* closed-loop RoboTwin environment runs in the ``robotwin`` env (which
pins a conflicting SAPIEN/torch). RoboTwin solves exactly this with a policy
server/client split; this mirrors it: run this server in the ``roboverse`` env, run
``eval_robotwin_policy.py --policy dp --server <host:port>`` in the ``robotwin`` env.

Protocol: length-prefixed (4-byte big-endian) pickled dicts over TCP.
  client -> {"cmd": "reset"}                                          -> {"ok": True}
  client -> {"cmd": "predict", "head_camera": HxWx3 uint8,
             "joint_qpos": (A,) float}                                -> {"action_chunk": (n_action, A) float}
  client -> {"cmd": "ping"}                                           -> {"ok": True, "n_action_steps": int}

Run (roboverse env, worktree on PYTHONPATH so roboverse_learn resolves locally)::

    PYTHONPATH=$PWD conda run -n roboverse python \\
        tools/robotwin_integration/dp_policy_server.py \\
        --ckpt il_outputs/ddpm_unet/beat_block_hammer/checkpoints/<ckpt> --port 5599
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys


def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv_msg(conn: socket.socket):
    header = _recv_exactly(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    body = _recv_exactly(conn, length)
    return None if body is None else pickle.loads(body)


def _send_msg(conn: socket.socket, obj) -> None:
    body = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(body)) + body)


class _DPInference:
    """Loads a DP checkpoint and runs the model's predict_action over an obs deque.

    Reuses RoboVerse's DefaultRunner (builds the model from the ckpt cfg) +
    DefaultEvalRunner (loads weights, maintains the n_obs_steps deque, calls
    policy.predict_action). The scenario-coupled action post-processing in
    DefaultEvalRunner.get_action is bypassed: we feed pre-formatted
    {head_cam, agent_pos} straight to predict_action and return the raw joint
    action chunk, so this runs with scenario=None (no RoboVerse env needed).
    """

    def __init__(self, ckpt: str, device: str = "cuda:0"):
        import dill
        import torch

        from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner
        from roboverse_learn.il.runners.default_runner import DefaultRunner

        self.torch = torch
        payload = torch.load(open(ckpt, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        self.n_action_steps = int(cfg.n_action_steps)
        self.device = device

        runner = DefaultRunner(cfg)  # builds the model architecture from the ckpt cfg
        # DefaultEvalRunner._init_policy re-reads the ckpt, loads weights into the
        # model, picks the EMA model when trained with EMA, and sets eval mode.
        self.eval_runner = DefaultEvalRunner(
            runner, scenario=None, num_envs=1, checkpoint_path=ckpt, device=device, task_name=cfg.task_name,
        )  # fmt: skip
        print(f"[dp_server] loaded {ckpt} (n_action_steps={self.n_action_steps}, device={device})", flush=True)

    def reset(self) -> None:
        self.eval_runner.reset()

    def predict(self, head_camera, joint_qpos):
        import numpy as np

        torch = self.torch
        # head_camera: HxWx3 uint8 -> (1, 3, H, W) float in [0,1]; agent_pos: (1, A).
        img = torch.from_numpy(np.ascontiguousarray(head_camera)).to(self.device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        agent_pos = torch.from_numpy(np.asarray(joint_qpos, dtype=np.float32)).to(self.device).unsqueeze(0)  # (1, A)
        processed = {"head_cam": img, "agent_pos": agent_pos}
        chunk = self.eval_runner.predict_action(processed)  # (n_action, num_envs, action_dim)
        return chunk[:, 0, :].detach().cpu().numpy()  # (n_action, action_dim)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="DP checkpoint (il_outputs/<policy>/<task>/checkpoints/<ckpt>)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    engine = _DPInference(args.ckpt, device=args.device)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[dp_server] listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"[dp_server] client {addr} connected", flush=True)
        try:
            while True:
                msg = _recv_msg(conn)
                if msg is None:
                    break
                cmd = msg.get("cmd")
                if cmd == "ping":
                    _send_msg(conn, {"ok": True, "n_action_steps": engine.n_action_steps})
                elif cmd == "reset":
                    engine.reset()
                    _send_msg(conn, {"ok": True})
                elif cmd == "predict":
                    chunk = engine.predict(msg["head_camera"], msg["joint_qpos"])
                    _send_msg(conn, {"action_chunk": chunk})
                else:
                    _send_msg(conn, {"error": f"unknown cmd {cmd!r}"})
        except Exception as e:  # keep the server alive across client errors
            print(f"[dp_server] client error: {type(e).__name__}: {e}", flush=True)
        finally:
            conn.close()
            print("[dp_server] client disconnected", flush=True)


if __name__ == "__main__":
    sys.exit(main())
