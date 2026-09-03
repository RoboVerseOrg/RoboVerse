"""openVLA-OFT policy server (runs in the `openvla` env, py3.10 + torch 2.7/cu128).

Half of the sim<->policy bridge that lets the official LIBERO-plus VLA drive the
LIBERO-plus simulator across the py3.8/py3.10 + sm_120 env split (the sim pins
py3.8/torch2.4.1 which can't use the RTX 5090; the 7B VLA needs torch>=2.7).

This process loads the official checkpoint ONCE and stays resident, serving
framed-pickle requests over a localhost socket:
  * {"cmd":"init"}                       -> {resize_size, chunk, num_steps_wait, dummy_action}
  * {"cmd":"act", "obs":..., "task":..., "unnorm_key":...} -> processed action chunk (np[chunk,7])

The image rotation / proprio assembly / gripper processing are replicated
*verbatim* from openvla-oft's experiments/robot/libero (we can't import that
module here because it pulls in `libero`, which lives only in the sim env), using
its own libero-free utilities so inference matches the official eval exactly.

Run:
    cd <path-to-openvla-oft checkout>
    python /path/to/bridge_policy_server.py --ckpt <ckpt_dir> --port 5555
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys

import numpy as np
import torch


def _recv(conn) -> dict:
    raw = b""
    (n,) = struct.unpack(">I", _recvn(conn, 4))
    return pickle.loads(_recvn(conn, n))


def _recvn(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def _send(conn, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


def build():
    """Load the openVLA-OFT model + heads + processor (official path).

    We can't import ``GenerateConfig`` from openvla-oft's ``run_libero_eval``
    because that module imports ``libero`` (sim-only). Replicate the config as a
    plain namespace with every field the openVLA loaders/inference read.
    """
    from types import SimpleNamespace

    if ARGS.quant != "none":
        # transformers 4.40 + new bitsandbytes: accelerate's dispatch_model calls
        # model.to(device) on the already-placed quantized model, which raises.
        # The move is redundant (bnb already placed it on the right device/dtype),
        # so make .to a no-op for quantized models instead of crashing.
        import transformers.modeling_utils as _mu

        _orig_to = _mu.PreTrainedModel.to

        def _safe_to(self, *a, **k):
            try:
                return _orig_to(self, *a, **k)
            except ValueError as e:
                if any(s in str(e) for s in ("4-bit", "8-bit", "bitsandbytes")):
                    return self
                raise

        _mu.PreTrainedModel.to = _safe_to

    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
    )
    from experiments.robot.robot_utils import get_image_resize_size

    cfg = SimpleNamespace(
        pretrained_checkpoint=ARGS.ckpt,
        model_family="openvla",
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        lora_rank=32,
        unnorm_key="libero_object",
        load_in_8bit=(ARGS.quant == "8bit"),
        load_in_4bit=(ARGS.quant == "4bit"),
        task_suite_name="libero_object",
        num_steps_wait=10,
        env_img_res=256,
        seed=7,
    )
    print(f"[server] loading checkpoint (quant={ARGS.quant}) ...", flush=True)
    model = get_vla(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    print(f"[server] model loaded; norm_stats keys = {list(model.norm_stats.keys())}", flush=True)
    return cfg, model, action_head, proprio_projector, processor, resize_size


def main() -> int:
    # openvla-oft's run_libero_eval imports `libero`; we can't. Pull its transforms
    # from the libero-free utilities, and inline quat2axisangle / invert_gripper_action
    # (verbatim from robosuite / openvla-oft) since their home modules import `libero`/tf.
    import math

    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.robot_utils import get_action, normalize_gripper_action
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    def quat2axisangle(quat):  # robosuite transform_utils.quat2axisangle, quat=(x,y,z,w)
        quat = np.asarray(quat, dtype=np.float64).copy()
        quat[3] = min(max(quat[3], -1.0), 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def invert_gripper_action(action):  # flip gripper sign (openvla dataloader convention)
        action = np.asarray(action).copy()
        action[..., -1] = action[..., -1] * -1.0
        return action

    cfg, model, action_head, proprio_projector, processor, resize_size = build()

    def prepare_observation(obs):
        # IMPORTANT: rotate 180 deg to match openvla-oft train preprocessing.
        img = np.asarray(obs["agentview_image"])[::-1, ::-1]
        wrist = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ])
        return {
            "full_image": resize_image_for_policy(img, resize_size),
            "wrist_image": resize_image_for_policy(wrist, resize_size),
            "state": state,
        }

    def process(action):
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)  # openvla family
        return action

    dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # get_libero_dummy_action (openvla)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", ARGS.port))
    srv.listen(1)
    print(f"[server] listening on 127.0.0.1:{ARGS.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[server] client connected", flush=True)
        try:
            while True:
                req = _recv(conn)
                cmd = req.get("cmd")
                if cmd == "init":
                    _send(
                        conn,
                        {
                            "resize_size": resize_size,
                            "chunk": int(NUM_ACTIONS_CHUNK),
                            "num_steps_wait": int(cfg.num_steps_wait),
                            "dummy_action": dummy,
                        },
                    )
                elif cmd == "act":
                    cfg.unnorm_key = req["unnorm_key"]
                    obs = req["obs"]
                    observation = prepare_observation(obs)
                    with torch.no_grad():
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            req["task"],
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            use_film=cfg.use_film,
                        )
                    chunk = np.stack([process(np.asarray(a)) for a in actions]).astype(np.float32)
                    _send(conn, {"actions": chunk})
                elif cmd == "bye":
                    break
                else:
                    _send(conn, {"error": f"unknown cmd {cmd}"})
        except ConnectionError:
            print("[server] client disconnected", flush=True)
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    ap.add_argument(
        "--repo",
        default=os.environ.get("OPENVLA_OFT_REPO", ""),
        help="path to a local openvla-oft checkout (or set $OPENVLA_OFT_REPO)",
    )
    ARGS = ap.parse_args()
    if not ARGS.repo:
        raise SystemExit("--repo (or $OPENVLA_OFT_REPO) must point to a local openvla-oft checkout")
    sys.path.insert(0, ARGS.repo)
    raise SystemExit(main())
