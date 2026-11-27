"""Off-the-shelf end-effector-delta VLAs as ``metasim.bench`` policies (the ``visual`` tier's generalist row).

A vision-language-action model of the OpenVLA / Octo / RT family maps one camera image and the
instruction to a 7-vector: a translation delta of the end effector, a rotation delta and a gripper
command. :class:`EEDeltaPolicy` turns that into the joint targets the harness steps, through the robot's
own kinematics (:class:`metasim.bench.skills.ArmModel`), and holds each model action for ``repeat``
simulator steps so a model trained at a few hertz is not asked for a new action every millisecond.
:class:`OpenVLAPolicy` is the concrete adapter for ``openvla/openvla-7b`` (bridge normalisation).

Run it from its own environment and let the harness talk to it over HTTP::

    python -m metasim bench serve --policy roboverse_learn.vla.bench_policy:OpenVLAPolicy --device cuda:0 --port 8100
    python -m metasim bench suite run --suite tabletop-mujoco --policy-url http://127.0.0.1:8100 \\
        --access visual --camera front

The adapter needs only the observation the ``visual`` tier leaves: the camera image and the robot's joint
positions. Zero-shot numbers on a robot and scene the model never saw are expected to be low; the row
exists so a fine-tuned checkpoint has a protocol-identical baseline to beat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from metasim.bench import PolicyContext, TaskInstance
from metasim.bench.skills import ArmModel, EnvView, _rot_about, env_view
from metasim.utils.log import warn_once


class EEDeltaPolicy:
    """Image + instruction -> ``(dx, dy, dz, droll, dpitch, dyaw, gripper)`` -> joint targets.

    Subclasses implement :meth:`predict` (one 7-vector per call). The deltas are applied to the tool
    centre point in the world frame: translation in metres (clipped to ``max_step`` per model action),
    rotation as roll / pitch / yaw increments in radians, gripper ``> 0.5`` = open.

    Args:
        camera: Camera to feed (default: the first the observation carries).
        repeat: Simulator steps each model action is held for.
        max_step: Clip on the translation delta per model action, metres.
        tcp_offset: Tool centre point in the end-effector frame (Franka default).
    """

    name = "ee-delta-vla"
    family = "vla"

    def __init__(
        self,
        camera: str | None = None,
        repeat: int = 4,
        max_step: float = 0.05,
        tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.10312),
    ) -> None:
        self.camera = camera
        self.repeat = max(1, int(repeat))
        self.max_step = float(max_step)
        self._tcp = tuple(float(v) for v in tcp_offset)
        self._arm: ArmModel | None = None
        self._robot: str | None = None
        self._prompt: dict[int, str | None] = {}
        self._held: dict[int, tuple[np.ndarray, float, int]] = {}
        self.calls = 0

    # ------------------------------------------------------------------------------------------ contract

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Remember each episode's instruction; nothing is held over from the previous one."""
        self._setup(ctx)
        for env_id, inst in zip(env_ids, instances, strict=True):
            self._prompt[env_id] = getattr(inst, "prompt", None) or ctx.instruction
            self._held.pop(env_id, None)

    def act(self, obs: Any, ctx: PolicyContext) -> list:
        """Joint targets for every env: a new model action every ``repeat`` steps, held in between."""
        self._setup(ctx)
        batch = []
        for env_id in range(ctx.num_envs):
            view = env_view(obs, env_id, self._robot, self._arm.joint_names)
            held = self._held.get(env_id)
            if held is None or held[2] <= 0:
                q_target, gripper = self._plan(view, obs, env_id)
                held = (q_target, gripper, self.repeat)
            q_target, gripper, left = held
            self._held[env_id] = (q_target, gripper, left - 1)
            targets = dict(zip(self._arm.arm_joints, (float(v) for v in q_target), strict=True))
            for j in self._arm.gripper_joints:
                targets[j] = gripper
            batch.append({self._robot: {"dof_pos_target": targets}})
        return batch

    def predict(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """The model's 7-vector for ``image`` (H, W, 3 uint8) and ``instruction``."""
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------- inner

    def _setup(self, ctx: PolicyContext) -> None:
        if self._arm is not None:
            return
        if len(ctx.robots) != 1:
            raise ValueError(f"{type(self).__name__} drives one robot; the context names {ctx.robots}")
        from metasim.utils.setup_util import get_robot

        self._robot = ctx.robots[0]
        self._arm = ArmModel(get_robot(self._robot), tcp_offset=self._tcp)
        if list(ctx.action_joint_names[self._robot]) != self._arm.joint_names:
            raise ValueError("action joint order differs from the robot config's")

    def _image(self, obs: Any, env_id: int) -> np.ndarray:
        cameras = getattr(obs, "cameras", None) or {}
        name = self.camera or next(iter(cameras), None)
        cam = cameras.get(name) if name else None
        rgb = getattr(cam, "rgb", None) if cam is not None else None
        if rgb is None:
            raise ValueError(
                f"{type(self).__name__} needs a camera image; the observation carries {sorted(cameras) or 'no camera'} "
                "(add one with --camera front)"
            )
        frame = rgb[env_id]
        frame = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(frame[..., :3])

    def _plan(self, view: EnvView, obs: Any, env_id: int) -> tuple[np.ndarray, float]:
        instruction = self._prompt.get(env_id) or ""
        if not instruction:
            warn_once((type(self), "no-instruction"), f"{type(self).__name__}: no instruction for env {env_id}")
        action = np.asarray(self.predict(self._image(obs, env_id), instruction), dtype=float).reshape(-1)
        self.calls += 1
        if action.shape[0] != 7:
            raise ValueError(f"{type(self).__name__}.predict returned {action.shape[0]} values, expected 7")
        q = np.array([view.q[j] for j in self._arm.arm_joints])
        pos_b, rot_b = self._arm.fk(q)
        pos, rot = view.base_to_world(pos_b, rot_b)
        delta = action[:3]
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step:
            delta = delta / norm * self.max_step
        roll, pitch, yaw = (float(v) for v in action[3:6])
        d_rot = _rot_about(np.array([0.0, 0.0, 1.0]), yaw) @ _rot_about(np.array([0.0, 1.0, 0.0]), pitch)
        d_rot = d_rot @ _rot_about(np.array([1.0, 0.0, 0.0]), roll)
        target_pos = pos + delta
        target_pos[2] = max(target_pos[2], 0.005)  # the table is at z = 0; never command the tool through it
        target_rot = d_rot @ rot
        p_b, r_b = view.world_to_base(target_pos, target_rot)
        q_target, _, _ = self._arm.ik(q, p_b, r_b, iters=30)
        gripper = self._arm.gripper_open if action[6] > 0.5 else self._arm.gripper_closed
        return q_target, gripper


class OpenVLAPolicy(EEDeltaPolicy):
    """``openvla/openvla-7b`` through ``transformers`` (``trust_remote_code``), bfloat16.

    Args:
        model: Hub id or local path.
        unnorm_key: Dataset statistics the model de-normalises with (``bridge_orig`` is the Bridge V2
            WidowX convention the released model was evaluated with).
        device: Torch device for the model.
        Other arguments: see :class:`EEDeltaPolicy`.
    """

    name = "openvla-7b"

    def __init__(
        self,
        model: str = "openvla/openvla-7b",
        unnorm_key: str = "bridge_orig",
        device: str = "cuda",
        camera: str | None = None,
        repeat: int = 4,
        max_step: float = 0.05,
        tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.10312),
    ) -> None:
        super().__init__(camera=camera, repeat=repeat, max_step=max_step, tcp_offset=tcp_offset)
        self.model_id = model
        self.unnorm_key = unnorm_key
        self.device = device
        self.name = f"openvla-7b[{unnorm_key}]"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """One OpenVLA action: translation (m), rotation (rad), gripper (1 open)."""
        from PIL import Image

        self._load()
        prompt = f"In: What action should the robot take to {instruction.rstrip('.').lower()}?\nOut:"
        inputs = self._processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        action = self._model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=float)
