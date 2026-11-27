"""A language-model agent on the benchmark's policy contract: plans with tool calls, acts through skills.

The agent is the ``agent-*`` tiers' reference policy. Each episode it receives the instruction, the
observation the tier allows (object poses under ``agent-privileged``; the camera image only under
``agent-skills``) and a tool list; every tool it calls is executed by :mod:`metasim.bench.skills`
closed-loop, and the outcome is returned to it before the next decision. Model calls, tokens and tool
calls are counted for the tier's budgets (``usage()``).

Backed by the Anthropic Messages API (``pip install anthropic``, ``ANTHROPIC_API_KEY`` in the
environment). No model call happens at import or construction, so the policy can be inspected and
tested without a key; ``act`` raises on the first episode when the key is missing.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from typing import Any

import numpy as np

from .policy import PolicyContext, _hold_dict_action
from .skills import (
    ArmModel,
    EnvView,
    Gripper,
    MoveTo,
    Skill,
    SkillRunner,
    env_view,
    pick,
    place_on,
    push_to,
    stand_upright,
    top_down_rot,
)

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You control a single-arm robot (a Franka Panda with a parallel gripper) on a table and must complete the
instruction below with as few tool calls as possible. Coordinates are world metres: the table top is z = 0, +z is up,
the robot base is at x = {base_x:.2f}, y = 0 and the workspace in front of it spans roughly x in [-0.2, 0.3],
y in [-0.3, 0.3]. The tool centre point (TCP) is the midpoint between the fingertips; a top-down grasp puts the TCP
at the object's centre. Cubes are 4 cm; the gripper opens to 8 cm.

{perception}

Call one tool at a time and read its result before the next call. Call `done` once the instruction is satisfied or
when you cannot make progress. Keep any text short."""

PERCEPTION_PRIVILEGED = (
    "You are told the pose of every object (centre position and yaw) and the TCP pose after each tool call; call "
    "`get_state` to read them again."
)
PERCEPTION_VISUAL = (
    "You see the scene only through the camera image attached to each decision ({cameras}); no object poses are "
    "given. Estimate positions from the image and the workspace description, then act with `move_to` / `grasp` / "
    "`release` / `lift` and check the next image."
)

TOOLS_STATE = [
    {
        "name": "get_state",
        "description": "Object poses (name, centre xyz, yaw in degrees) and the TCP pose.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pick",
        "description": "Top-down grasp of a named object and a lift of 12 cm.",
        "input_schema": {
            "type": "object",
            "properties": {"object": {"type": "string"}},
            "required": ["object"],
        },
    },
    {
        "name": "place_on",
        "description": "Carry the held object above a named object and set it down on top of it, then release.",
        "input_schema": {
            "type": "object",
            "properties": {"object": {"type": "string"}},
            "required": ["object"],
        },
    },
    {
        "name": "push",
        "description": "Push a named object along the table until its centre is at a target: another object's position or an xy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "target_object": {"type": "string"},
                "target_xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["object"],
        },
    },
    {
        "name": "stand_upright",
        "description": "Grasp a lying bar-shaped object at its centre, lift it, and stand it vertically on one end.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "length": {"type": "number", "description": "bar length in metres"},
            },
            "required": ["object", "length"],
        },
    },
]
TOOLS_MOTION = [
    {
        "name": "move_to",
        "description": "Move the TCP to a world position with a top-down tool pointing at the table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "yaw_deg": {"type": "number", "description": "rotation of the fingers about z, degrees (default 0)"},
            },
            "required": ["x", "y", "z"],
        },
    },
    {"name": "grasp", "description": "Close the gripper.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "release", "description": "Open the gripper.", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "lift",
        "description": "Raise the TCP straight up by dz metres.",
        "input_schema": {"type": "object", "properties": {"dz": {"type": "number"}}, "required": ["dz"]},
    },
    {
        "name": "done",
        "description": "Stop acting; the robot holds its pose until the episode ends.",
        "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}},
    },
]


class _Episode:
    """Per-env agent state: the transcript, the skill runner and whether the agent has stopped."""

    def __init__(self, runner: SkillRunner, instruction: str | None) -> None:
        self.runner = runner
        self.instruction = instruction
        self.messages: list[dict[str, Any]] = []
        self.pending_tool: dict[str, Any] | None = None
        self.finished = False
        self.calls = 0
        self.decisions: list[str] = []


class LLMAgentPolicy:
    """The LLM agent baseline (``family = "agent"``): plans with tool calls executed by the scripted skills.

    Args:
        model: Anthropic model id.
        max_calls: Model calls per episode; the agent holds once they are spent.
        camera: Camera to attach to each decision (default: the first camera the observation carries).
        tcp_offset: Tool centre point in the end-effector frame (Franka default).
    """

    name = "llm-agent"
    family = "agent"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_calls: int = 8,
        camera: str | None = None,
        tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.10312),
    ) -> None:
        self.model = model
        self.max_calls = int(max_calls)
        self.camera = camera
        self._tcp = tuple(float(v) for v in tcp_offset)
        self._client = None
        self._arm: ArmModel | None = None
        self._robot: str | None = None
        self._episodes: dict[int, _Episode] = {}
        self._pending_reset: dict[int, str | None] = {}
        self._usage = {"llm_calls": 0, "llm_tokens": 0, "tool_calls": 0, "llm_time_s": 0.0}
        self.transcripts: list[list[str]] = []

    # ----------------------------------------------------------------------------------------- contract

    def reset(self, env_ids: list[int], instances, ctx: PolicyContext) -> None:
        """Start a fresh transcript per env; the first decision is made on the first observation."""
        self._setup(ctx)
        for env_id, inst in zip(env_ids, instances, strict=True):
            old = self._episodes.pop(env_id, None)
            if old is not None and old.decisions:
                self.transcripts.append(old.decisions)
            self._pending_reset[env_id] = getattr(inst, "prompt", None) or ctx.instruction

    def act(self, obs: Any, ctx: PolicyContext) -> list:
        """One step per env: run the active skill, or ask the model for the next tool when none is active."""
        self._setup(ctx)
        batch = []
        hold = None
        for env_id in range(ctx.num_envs):
            if env_id in self._pending_reset:
                instruction = self._pending_reset.pop(env_id)
                self._episodes[env_id] = _Episode(SkillRunner(self._arm, self._robot), instruction)
            ep = self._episodes.get(env_id)
            if ep is None:
                hold = hold or _hold_dict_action(ctx)
                batch.append(hold)
                continue
            view = env_view(obs, env_id, self._robot, self._arm.joint_names)
            if ep.runner.done and ep.pending_tool is not None:  # the tool just finished: report back
                ep.messages.append(self._tool_result(ep, view))
                ep.pending_tool = None
            if ep.runner.done and not ep.finished:
                self._decide(ep, view, obs, env_id, ctx)
            batch.append(ep.runner.act(view))
        return batch

    def usage(self) -> dict[str, float]:
        """Model calls, tokens and tool calls over the run (read by the evaluator for the tier budgets)."""
        return dict(self._usage)

    # ------------------------------------------------------------------------------------------ set-up

    def _setup(self, ctx: PolicyContext) -> None:
        if self._arm is not None:
            return
        access = ctx.access or {}
        if access.get("tools", "sim_queries") == "none":
            raise ValueError(f"access tier {access.get('name')!r} allows no tools; the agent needs skills")
        scenario = ctx.extras.get("scenario")
        robots = list(getattr(scenario, "robots", []) or [])
        if len(robots) != 1:
            raise ValueError(f"LLMAgentPolicy drives one robot; the scenario has {len(robots)}")
        self._robot = robots[0].name
        self._arm = ArmModel(robots[0], tcp_offset=self._tcp)
        from . import skills as _skills

        _skills._CURRENT_ARM[self._robot] = self._arm
        self._privileged = access.get("observation", "privileged") == "privileged"
        self._cameras = [c.name for c in (getattr(scenario, "cameras", None) or [])]
        allowed = access.get("cameras")
        if allowed is not None:
            self._cameras = [c for c in self._cameras if c in allowed]
        self._tools = (TOOLS_STATE if self._privileged else []) + TOOLS_MOTION
        self._system = SYSTEM_PROMPT.format(
            base_x=-0.615,
            perception=PERCEPTION_PRIVILEGED
            if self._privileged
            else PERCEPTION_VISUAL.format(cameras=", ".join(self._cameras) or "no camera"),
        )

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as err:
                raise ImportError("LLMAgentPolicy needs the `anthropic` package (pip install anthropic)") from err
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("LLMAgentPolicy needs ANTHROPIC_API_KEY in the environment")
            self._client = anthropic.Anthropic()
        return self._client

    # ---------------------------------------------------------------------------------------- deciding

    def _decide(self, ep: _Episode, view: EnvView, obs: Any, env_id: int, ctx: PolicyContext) -> None:
        """Ask the model for the next tool call and load its skills; ``done`` or the call budget ends acting."""
        if ep.calls >= self.max_calls:
            ep.finished = True
            ep.decisions.append(f"budget: {self.max_calls} calls spent")
            return
        if not ep.messages:
            ep.messages.append({"role": "user", "content": self._first_turn(ep, view, obs, env_id)})
        reply = self._call(ep)
        tool_use = next((b for b in reply if b.get("type") == "tool_use"), None)
        text = " ".join(b.get("text", "") for b in reply if b.get("type") == "text").strip()
        if tool_use is not None:  # one tool per turn: later tool_use blocks would need results we never produce
            reply = reply[: reply.index(tool_use) + 1]
        ep.messages.append({"role": "assistant", "content": reply})
        if tool_use is None:
            ep.finished = True
            ep.decisions.append(f"stop (no tool): {text[:120]}")
            return
        name, args = tool_use["name"], dict(tool_use.get("input") or {})
        ep.decisions.append(f"{name}({json.dumps(args, sort_keys=True)})" + (f"  # {text[:80]}" if text else ""))
        self._usage["tool_calls"] += 1
        if name == "done":
            ep.finished = True
            ep.messages.append(_tool_result_message(tool_use["id"], "stopped"))
            return
        if name == "get_state":
            ep.messages.append(_tool_result_message(tool_use["id"], self._describe(view)))
            self._decide(ep, view, obs, env_id, ctx)  # a query costs no sim steps; decide again
            return
        try:
            skills = self._skills_for(name, args, view)
        except (KeyError, ValueError, TypeError) as err:
            ep.messages.append(_tool_result_message(tool_use["id"], f"error: {err}", is_error=True))
            self._decide(ep, view, obs, env_id, ctx)
            return
        ep.runner.load(skills)
        ep.pending_tool = tool_use

    def _skills_for(self, name: str, args: dict[str, Any], view: EnvView) -> list[Skill]:
        arm = self._arm
        if name == "pick":
            return pick(str(args["object"]), arm, lift=0.12)
        if name == "place_on":
            return place_on(str(args["object"]), arm)
        if name == "push":
            target: Any = args.get("target_object") or tuple(float(v) for v in args.get("target_xy") or ())
            if not target:
                raise ValueError("push needs target_object or target_xy")
            return push_to(str(args["object"]), target, arm)
        if name == "stand_upright":
            return stand_upright(str(args["object"]), arm, length=float(args["length"]), width=0.05)
        if name == "move_to":
            pos = np.array([float(args["x"]), float(args["y"]), float(args["z"])])
            rot = top_down_rot(math.radians(float(args.get("yaw_deg", 0.0))))
            gripper = float(view.q[arm.gripper_joints[0]]) if arm.gripper_joints else 0.0
            return [MoveTo((pos, rot), gripper, name="move_to")]
        if name == "grasp":
            return [Gripper(arm.gripper_closed, name="grasp")]
        if name == "release":
            return [Gripper(arm.gripper_open, name="release")]
        if name == "lift":
            from .skills import _offset_current

            gripper = float(view.q[arm.gripper_joints[0]]) if arm.gripper_joints else 0.0
            return [MoveTo(_offset_current(0.0, 0.0, float(args["dz"])), gripper, name="lift")]
        raise ValueError(f"unknown tool {name!r}")

    def _first_turn(self, ep: _Episode, view: EnvView, obs: Any, env_id: int) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": f"Instruction: {ep.instruction or '(none given)'}"}]
        if self._privileged:
            content.append({"type": "text", "text": self._describe(view)})
        image = self._image(obs, env_id)
        if image is not None:
            content.append(image)
        return content

    def _tool_result(self, ep: _Episode, view: EnvView) -> dict[str, Any]:
        log = ep.runner.log
        stalled = any("stalled" in line for line in log)
        text = f"executed {ep.pending_tool['name']}: {', '.join(log)}."
        if stalled:
            text += " Some motion did not reach its target (stalled)."
        if self._privileged:
            text += "\n" + self._describe(view)
        return _tool_result_message(ep.pending_tool["id"], text)

    def _describe(self, view: EnvView) -> str:
        from .skills import object_yaw

        q = np.array([view.q[j] for j in self._arm.arm_joints])
        pos_b, rot_b = self._arm.fk(q)
        tcp, _ = view.base_to_world(pos_b, rot_b)
        opening = float(view.q[self._arm.gripper_joints[0]]) if self._arm.gripper_joints else 0.0
        lines = [f"TCP at ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}), gripper opening {2 * opening:.3f} m."]
        for name, (pos, quat) in sorted(view.objects.items()):
            lines.append(
                f"{name}: centre ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), yaw {math.degrees(object_yaw(quat)):.0f} deg"
            )
        return "\n".join(lines)

    def _image(self, obs: Any, env_id: int) -> dict[str, Any] | None:
        cameras = getattr(obs, "cameras", None) or {}
        name = self.camera or next(iter(cameras), None)
        cam = cameras.get(name) if name else None
        rgb = getattr(cam, "rgb", None) if cam is not None else None
        if rgb is None:
            return None
        from PIL import Image

        frame = rgb[env_id].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(frame[..., :3]).save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    def _call(self, ep: _Episode) -> list[dict[str, Any]]:
        client = self._client_or_raise()
        started = time.perf_counter()
        response = client.messages.create(
            model=self.model,
            max_tokens=600,
            system=self._system,
            tools=self._tools,
            messages=ep.messages,
        )
        self._usage["llm_time_s"] += time.perf_counter() - started
        self._usage["llm_calls"] += 1
        ep.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage["llm_tokens"] += int(getattr(usage, "input_tokens", 0) or 0) + int(
                getattr(usage, "output_tokens", 0) or 0
            )
        blocks = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return blocks


def _tool_result_message(tool_use_id: str, text: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    if is_error:
        block["is_error"] = True
    return {"role": "user", "content": [block]}
