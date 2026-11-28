"""Policies served over HTTP: score a policy that lives in another process, venv or machine.

VLA and imitation stacks pin their own torch / transformers / jax and rarely install next to a
simulator. A :class:`PolicyServer` wraps any :class:`~metasim.bench.Policy` in that policy's own
environment; :class:`RemotePolicy` is the evaluator-side stand-in that forwards ``reset`` / ``act``
to it. The harness keeps the loop, the scoring and the result, exactly as for an in-process policy.

Wire protocol ``roboverse.bench.remote/1`` (stdlib HTTP + NumPy, no pickle anywhere):

* ``GET  <url>/metadata`` → JSON ``{"protocol", "name", ...}``; whatever else the server reports
  (``checkpoint``, ``checkpoint_sha256``) is written into the result's policy record, and
  ``required_metadata`` makes a mismatch fail before the first episode.
* ``POST <url>/reset`` JSON ``{"context": {...}, "env_ids": [...], "instances": [...]}`` — the
  :class:`~metasim.bench.PolicyContext` (minus ``extras``) and, per starting env, the instance's
  index, seed, id and prompt.
* ``POST <url>/act`` NPZ (``application/octet-stream``) with every array leaf of the observation under
  ``obs/<path>`` (``obs/robots/franka/joint_pos``, ``obs/cameras/front/rgb``), ``prompt`` (one string per
  env, ``""`` for none), ``request`` and ``task``. The response is JSON ``{"actions": ...}`` of shape
  ``(num_envs, action_dim)``, or ``(num_envs, horizon, action_dim)`` for an action chunk.

Action chunks: the client executes each env's chunk step by step. When an env has run out (its
chunk is used up, or it was reset), the client asks the server for the whole batch and hands the new
rows only to the envs that ran out; every other env finishes the chunk it is executing. So each env
executes every chunk it receives to the end, as it would with ``num_envs=1``. A policy that keeps
per-env state across calls must key it by env (``reset`` names the envs that start). On the server,
observations arrive as nested mappings of tensors with attribute access
(``obs.robots["franka"].joint_pos``), so a policy written against ``TensorState`` runs unchanged as
long as it reads tensors only. Mapping keys arrive as strings.
"""

from __future__ import annotations

import dataclasses
import http.client
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from loguru import logger as log

from .instances import TaskInstance
from .policy import Policy, PolicyContext

PROTOCOL = "roboverse.bench.remote/1"
MAX_BODY_BYTES = 1 << 30
"""Largest request the server reads (1 GiB: a few hundred envs of 480p RGB)."""


# --------------------------------------------------------------------------- observation codec


def flatten_observation(obs: Any, prefix: str = "obs", keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Every array leaf of ``obs`` as ``{"obs/<path>": ndarray}``.

    Tensors, NumPy arrays and numbers are leaves (half and bfloat16 tensors are sent as float32);
    mappings and dataclasses (``TensorState``, ``RobotState``, ``CameraState``) are walked; strings and
    ``None`` are skipped, and so are lists of strings (``body_names``). A list or tuple of values, an
    array of strings or objects and a key containing ``/`` raise ``ValueError`` rather than vanish on
    the way to the policy.
    ``keys`` keeps only paths that start with one of the given prefixes (``("robots", "cameras/front/rgb")``),
    which is how a client avoids shipping cameras the policy does not read.
    """
    out: dict[str, np.ndarray] = {}

    def wanted(path: str) -> bool:
        if keys is None or path == prefix:
            return True
        rel = path[len(prefix) + 1 :]
        return any(rel == k or rel.startswith(k + "/") or k.startswith(rel + "/") for k in keys)

    def walk(value: Any, path: str) -> None:
        if not wanted(path):
            return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.dtype in (torch.bfloat16, torch.float16):
                value = value.float()
            out[path] = value.cpu().numpy()
            return
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "OUSV":
                raise ValueError(
                    f"observation leaf {path} is a {value.dtype} array; only numeric arrays cross the wire"
                )
            out[path] = value
            return
        if isinstance(value, (bool, int, float, np.number)):
            out[path] = np.asarray(value)
            return
        if isinstance(value, dict):
            for k, v in value.items():
                if "/" in str(k):
                    raise ValueError(f"observation key {k!r} under {path} contains '/', the wire's path separator")
                walk(v, f"{path}/{k}")
            return
        if isinstance(value, (list, tuple)):
            if all(isinstance(v, str) for v in value):
                return  # names (``body_names``, ``joint_names``) are metadata, like a plain string
            raise ValueError(
                f"observation leaf {path} is a {type(value).__name__} of values; send a tensor or array (or filter it "
                "out with obs_keys)"
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                walk(getattr(value, f.name), f"{path}/{f.name}")

    walk(obs, prefix)
    return out


class ObsTree(dict):
    """A nested mapping that also answers attribute access (``tree.robots["franka"].joint_pos``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err


def unflatten_observation(arrays: dict[str, np.ndarray], prefix: str = "obs", device: str = "cpu") -> Any:
    """Inverse of :func:`flatten_observation`: an :class:`ObsTree` of tensors on ``device``."""
    root = ObsTree()
    for path, array in arrays.items():
        if path != prefix and not path.startswith(prefix + "/"):
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        parts = path.split("/")[1:]
        if not parts:
            return tensor
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, ObsTree())
        node[parts[-1]] = tensor
    return root


def _context_to_json(ctx: PolicyContext) -> dict[str, Any]:
    return {
        "task": ctx.task,
        "simulator": ctx.simulator,
        "robots": list(ctx.robots),
        "num_envs": ctx.num_envs,
        "seed": ctx.seed,
        "max_episode_steps": ctx.max_episode_steps,
        "action_joint_names": {k: list(v) for k, v in ctx.action_joint_names.items()},
        "action_low": ctx.action_low.tolist(),
        "action_high": ctx.action_high.tolist(),
        "default_action": ctx.default_action.tolist(),
        "default_joint_positions": ctx.default_joint_positions,
        "instruction": ctx.instruction,
        "access": ctx.access,
    }


def _context_from_json(data: dict[str, Any], device: str) -> PolicyContext:
    dev = torch.device(device)

    def vec(key: str) -> torch.Tensor:
        return torch.tensor([float(v) for v in data[key]], dtype=torch.float32, device=dev)

    return PolicyContext(
        task=str(data["task"]),
        simulator=str(data["simulator"]),
        robots=list(data["robots"]),
        num_envs=int(data["num_envs"]),
        device=dev,
        seed=int(data["seed"]),
        max_episode_steps=int(data["max_episode_steps"]),
        action_joint_names={k: list(v) for k, v in data["action_joint_names"].items()},
        action_low=vec("action_low"),
        action_high=vec("action_high"),
        default_action=vec("default_action"),
        default_joint_positions=data.get("default_joint_positions") or {},
        instruction=data.get("instruction"),
        access=data.get("access"),
    )


def _instance_to_json(inst: TaskInstance) -> dict[str, Any]:
    return {
        "index": inst.index,
        "seed": inst.seed,
        "instance_id": inst.instance_id,
        "prompt": inst.prompt,
        "prompt_variant": inst.prompt_variant,
    }


def actions_to_array(actions: Any, ctx: PolicyContext) -> np.ndarray:
    """A policy's ``act`` output as ``(num_envs, action_dim)`` floats in the handler's action order.

    Accepts a tensor / array, or the per-env ``{robot: {"dof_pos_target": {joint: value}}}`` batch;
    a joint a dict action leaves out gets its default position.
    """
    if isinstance(actions, torch.Tensor):
        return actions.detach().cpu().to(torch.float32).numpy()
    if isinstance(actions, np.ndarray):
        return actions.astype(np.float32)
    rows = []
    default = ctx.default_action.cpu().tolist()
    for env_action in actions:
        row, offset = [], 0
        for robot, joints in ctx.action_joint_names.items():
            targets = (env_action.get(robot) or {}).get("dof_pos_target") or {}
            for j, name in enumerate(joints):
                value = targets.get(name)
                row.append(float(default[offset + j]) if value is None else float(value))
            offset += len(joints)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- client


class RemotePolicy:
    """Evaluator-side proxy for a policy behind a :class:`PolicyServer` (or any server of the protocol).

    Args:
        url: Base URL, e.g. ``http://127.0.0.1:8000`` (``/metadata``, ``/reset``, ``/act`` are appended).
        required_metadata: Keys the server's metadata must report with these exact values (a checkpoint
            identity, say); a mismatch raises before the first episode.
        obs_keys: Observation path prefixes to send (default: everything).
        timeout: Seconds per request.
        retries: Re-tries on connection errors (not on HTTP errors, which are the server's verdict).
    """

    def __init__(
        self,
        url: str,
        *,
        required_metadata: dict[str, Any] | None = None,
        obs_keys: tuple[str, ...] | list[str] | None = None,
        timeout: float = 180.0,
        retries: int = 3,
        retry_wait_s: float = 2.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_wait_s = float(retry_wait_s)
        self.obs_keys = tuple(obs_keys) if obs_keys else None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
        self.metadata: dict[str, Any] = self._request("GET", "/metadata")
        self.family = self.metadata.get("family")  # the served policy's method family, recorded like a local one
        if self.metadata.get("protocol") != PROTOCOL:
            raise ValueError(f"{self.url} speaks {self.metadata.get('protocol')!r}, this client speaks {PROTOCOL!r}")
        mismatched = {
            k: (v, self.metadata.get(k)) for k, v in (required_metadata or {}).items() if self.metadata.get(k) != v
        }
        if mismatched:
            raise ValueError(f"policy server metadata mismatch (expected, got): {mismatched}")
        self.name = str(self.metadata.get("name") or "remote")
        self._chunks: dict[int, list[np.ndarray]] = {}
        self._prompts: dict[int, str] = {}
        self._requests = 0
        self._active: set[int] = set()

    def _request(self, method: str, path: str, body: bytes | None = None, content_type: str = "application/json"):
        url = self.url + path
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": content_type})
                with self._opener.open(req, timeout=self.timeout) as response:
                    raw = response.read().decode(errors="replace")
                try:
                    return json.loads(raw)
                except ValueError as err:
                    raise ValueError(f"policy server {method} {url} answered with non-JSON: {raw[:200]!r}") from err
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:2000]
                raise RuntimeError(f"policy server {method} {url} answered {err.code}: {detail}") from err
            except (urllib.error.URLError, ConnectionError, http.client.HTTPException, TimeoutError) as err:
                reason = getattr(err, "reason", err)
                if method != "GET" and not isinstance(reason, ConnectionRefusedError):
                    # the server may already have run this reset/act; replaying it would step a stateful policy twice
                    raise ConnectionError(f"policy server {method} {url} failed mid-request: {err}") from err
                if attempt == self.retries:
                    raise ConnectionError(
                        f"policy server {url} unreachable after {attempt + 1} attempt(s): {err}"
                    ) from err
                log.warning(f"bench remote: {method} {path} failed ({err}); retry {attempt + 1}/{self.retries}")
                time.sleep(self.retry_wait_s)
        raise AssertionError("unreachable")

    def reset(self, env_ids: list[int], instances: list[TaskInstance], ctx: PolicyContext) -> None:
        """Forward the episode starts; drop those envs' unexecuted chunk remainders."""
        payload = {
            "context": _context_to_json(ctx),
            "env_ids": list(env_ids),
            "instances": [_instance_to_json(i) for i in instances],
        }
        self._request("POST", "/reset", json.dumps(payload).encode())
        for env_id, inst in zip(env_ids, instances, strict=True):
            self._chunks[env_id] = []
            self._prompts[env_id] = inst.prompt or ""
            self._active.add(env_id)

    def act(self, obs: Any, ctx: PolicyContext) -> torch.Tensor:
        """The next action of every env, asking the server when a running env's chunk is used up."""
        empty = [i for i in sorted(self._active) if not self._chunks.get(i)]
        if empty:
            self._refill(obs, ctx, empty)
        rows = []
        for env_id in range(ctx.num_envs):
            chunk = self._chunks.get(env_id)
            rows.append(chunk.pop(0) if chunk else ctx.default_action.cpu().numpy())
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=ctx.device)

    def _refill(self, obs: Any, ctx: PolicyContext, env_ids: list[int]) -> None:
        arrays = flatten_observation(obs, keys=self.obs_keys)
        arrays["prompt"] = np.asarray([self._prompts.get(i, "") for i in range(ctx.num_envs)], dtype=str)
        arrays["request"] = np.asarray(self._requests)
        arrays["task"] = np.asarray(ctx.task, dtype=str)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        reply = self._request("POST", "/act", buf.getvalue(), content_type="application/octet-stream")
        self._requests += 1
        if not isinstance(reply, dict) or "actions" not in reply:
            raise ValueError(f"policy server {self.url}/act answered without 'actions': {str(reply)[:200]!r}")
        try:
            actions = np.asarray(reply["actions"], dtype=np.float32)
        except (TypeError, ValueError) as err:
            raise ValueError(f"policy server {self.url}/act: 'actions' is not a numeric array: {err}") from err
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[0] != ctx.num_envs or actions.shape[2] != ctx.action_dim:
            raise ValueError(
                f"policy server returned actions of shape {tuple(actions.shape)}; expected "
                f"({ctx.num_envs}, {ctx.action_dim}) or ({ctx.num_envs}, horizon, {ctx.action_dim})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy server returned non-finite actions")
        for i in env_ids:
            self._chunks[i] = list(actions[i])


# --------------------------------------------------------------------------- server


class PolicyServer:
    """Serve ``policy`` on ``host:port`` for :class:`RemotePolicy` clients.

    Connections are handled on threads (a stalled client cannot block the others, and is dropped
    after 30 s of silence), while the policy itself is called under one lock, one ``reset`` / ``act``
    at a time.

    Args:
        policy: Any :class:`Policy`. It sees a :class:`PolicyContext` rebuilt from the client's (with
            empty ``extras``: the simulator's scenario does not cross the wire) and instances carrying
            index, seed and prompt but no state or demo actions.
        host: Interface to bind; the default ``127.0.0.1`` is reachable from this machine only.
        port: TCP port; 0 picks a free one (read it back from :attr:`port`).
        device: Torch device the observations are moved to before ``policy.act``.
        metadata: Extra fields for ``GET /metadata`` (``checkpoint``, ``checkpoint_sha256``).
    """

    def __init__(
        self,
        policy: Policy,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        device: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.device = device
        self.metadata = {
            "protocol": PROTOCOL,
            "name": getattr(policy, "name", type(policy).__name__),
            **({"family": policy.family} if isinstance(getattr(policy, "family", None), str) else {}),
            **(metadata or {}),
        }
        self.ctx: PolicyContext | None = None
        server = self

        class _Handler(BaseHTTPRequestHandler):
            timeout = 30  # a client that stalls mid-request is dropped instead of holding a thread forever

            def log_message(self, fmt, *args):  # route through loguru at debug level
                log.debug("bench serve: " + fmt % args)

            def _reply(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                try:
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):  # the client is gone; nothing to tell it
                    self.close_connection = True

            def do_GET(self):
                if self.path.rstrip("/").endswith("/metadata"):
                    self._reply(200, server.metadata)
                else:
                    self._reply(404, {"error": f"unknown path {self.path}"})

            def do_POST(self):
                try:
                    length = int(self.headers["Content-Length"])
                except (KeyError, TypeError, ValueError):
                    self._reply(411, {"error": "POST needs a numeric Content-Length"})
                    return
                if length < 0 or length > MAX_BODY_BYTES:
                    self._reply(413, {"error": f"request length {length} is outside [0, {MAX_BODY_BYTES}]"})
                    return
                try:
                    body = self.rfile.read(length)
                except (TimeoutError, OSError):  # stalled past the handler timeout, or reset
                    self.close_connection = True
                    return
                if len(body) != length:  # the client closed before sending what it announced
                    self.close_connection = True
                    return
                route = self.path.rstrip("/")
                if not (route.endswith("/reset") or route.endswith("/act")):
                    self._reply(404, {"error": f"unknown path {self.path}"})
                    return
                try:
                    with server._lock:  # one policy call at a time; /metadata stays answerable meanwhile
                        if route.endswith("/reset"):
                            try:
                                payload = json.loads(body.decode())
                            except (ValueError, UnicodeDecodeError) as err:
                                self._reply(400, {"error": f"/reset body is not JSON: {err}"})
                                return
                            reply = server.handle_reset(payload)
                        else:
                            try:
                                with np.load(io.BytesIO(body), allow_pickle=False) as data:
                                    arrays = {k: data[k] for k in data.files}
                            except (ValueError, OSError, KeyError) as err:  # not an NPZ, or one with pickled objects
                                self._reply(400, {"error": f"/act body is not an NPZ of plain arrays: {err}"})
                                return
                            reply = server.handle_act(arrays)
                    self._reply(200, reply)
                except Exception as err:  # the client gets the reason instead of a dropped connection
                    log.exception("bench serve: request failed")
                    self._reply(500, {"error": f"{type(err).__name__}: {err}"})

        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self.host, self.port = self._httpd.server_address[:2]

    @property
    def url(self) -> str:
        """Base URL clients connect to."""
        return f"http://{self.host}:{self.port}"

    def handle_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /reset``: rebuild the context and call ``policy.reset``."""
        self.ctx = _context_from_json(payload["context"], self.device)
        instances = [
            TaskInstance(
                index=int(i["index"]),
                seed=int(i["seed"]),
                prompt=i.get("prompt"),
                prompt_variant=i.get("prompt_variant"),
            )
            for i in payload["instances"]
        ]
        self.policy.reset([int(e) for e in payload["env_ids"]], instances, self.ctx)
        return {"ok": True}

    def handle_act(self, body: bytes | dict[str, np.ndarray]) -> dict[str, Any]:
        """``POST /act``: decode the observation (NPZ bytes, or the arrays already read), call ``policy.act``, return the actions as JSON."""
        if self.ctx is None:
            raise RuntimeError("act before reset: the client must POST /reset first")
        if isinstance(body, dict):
            arrays = body
        else:
            with np.load(io.BytesIO(body), allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        obs = unflatten_observation(arrays, device=self.device)
        return {"actions": actions_to_array(self.policy.act(obs, self.ctx), self.ctx).tolist()}

    def serve_forever(self) -> None:
        """Block serving requests until :meth:`shutdown` (or Ctrl-C)."""
        log.info(f"bench serve: {self.metadata['name']} on {self.url} ({PROTOCOL})")
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def shutdown(self) -> None:
        """Stop :meth:`serve_forever` from another thread."""
        self._httpd.shutdown()


# --------------------------------------------------------------------------- conformance


@dataclasses.dataclass
class ConformanceRow:
    """One protocol check against a running server: pass / fail and what was observed."""

    check: str
    ok: bool
    detail: str


@dataclasses.dataclass
class ConformanceReport:
    """``roboverse.bench.remote/1`` conformance of one server, without a simulator."""

    url: str
    rows: list[ConformanceRow] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Every check passed."""
        return all(r.ok for r in self.rows)

    def table(self) -> str:
        """The report as a Markdown table."""
        lines = [f"policy server {self.url} ({PROTOCOL})", "| check | result | detail |", "|---|---|---|"]
        lines += [f"| {r.check} | {'ok' if r.ok else 'FAIL'} | {r.detail} |" for r in self.rows]
        lines.append(f"verdict: {'pass' if self.ok else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "url": self.url,
            "protocol": PROTOCOL,
            "ok": self.ok,
            "metadata": self.metadata,
            "rows": [dataclasses.asdict(r) for r in self.rows],
        }


def synthetic_context(
    *, num_envs: int = 2, robot_cfg: Any = None, action_dim: int = 7, task: str = "conformance"
) -> PolicyContext:
    """A :class:`PolicyContext` for the conformance run: ``robot_cfg``'s action box when given, else ``action_dim`` joints in [-1, 1]."""
    if robot_cfg is not None:
        joints = sorted(robot_cfg.actuators or {})
        limits = robot_cfg.joint_limits or {}
        defaults = robot_cfg.default_joint_positions or {}
        low = torch.tensor([float(limits.get(j, (-1.0, 1.0))[0]) for j in joints])
        high = torch.tensor([float(limits.get(j, (-1.0, 1.0))[1]) for j in joints])
        default = torch.tensor([float(defaults.get(j, 0.0)) for j in joints])
        robots = [str(robot_cfg.name)]
        names = {robots[0]: joints}
        default_joint_positions = {robots[0]: {j: float(defaults.get(j, 0.0)) for j in joints}}
    else:
        names = {"robot": [f"joint{i}" for i in range(int(action_dim))]}
        robots = ["robot"]
        low, high = -torch.ones(int(action_dim)), torch.ones(int(action_dim))
        default = torch.zeros(int(action_dim))
        default_joint_positions = {"robot": {j: 0.0 for j in names["robot"]}}
    return PolicyContext(
        task=task,
        simulator="none",
        robots=robots,
        num_envs=int(num_envs),
        device=torch.device("cpu"),
        seed=0,
        max_episode_steps=50,
        action_joint_names=names,
        action_low=low,
        action_high=high,
        default_action=default,
        default_joint_positions=default_joint_positions,
        instruction="conformance run: hold still",
        access=None,
    )


def synthetic_observation(ctx: PolicyContext, *, image_hw: tuple[int, int] = (64, 64)) -> dict[str, np.ndarray]:
    """The flattened observation the conformance run POSTs: joints at the default action, one black RGB camera."""
    n = ctx.num_envs
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for robot, joints in ctx.action_joint_names.items():
        q = ctx.default_action[offset : offset + len(joints)].cpu().numpy().astype(np.float32)
        arrays[f"obs/robots/{robot}/joint_pos"] = np.repeat(q[None], n, axis=0)
        arrays[f"obs/robots/{robot}/joint_vel"] = np.zeros((n, len(joints)), dtype=np.float32)
        offset += len(joints)
    arrays["obs/cameras/front/rgb"] = np.zeros((n, *image_hw, 3), dtype=np.uint8)
    return arrays


def conformance(
    url: str, *, num_envs: int = 2, robot_cfg: Any = None, action_dim: int = 7, timeout: float = 60.0
) -> ConformanceReport:
    """Exercise a server of ``roboverse.bench.remote/1`` the way the evaluator will, and report each contract.

    No simulator is involved: the context, the instances and the observation are synthetic. Checks: ``/metadata``
    speaks the protocol and names the policy; ``/reset`` accepts the context and instances; ``/act`` returns finite
    actions of shape ``(num_envs, action_dim)`` or ``(num_envs, horizon, action_dim)`` (the horizon is reported);
    a reset of one env followed by ``/act`` still answers for every env; a malformed ``/act`` body is refused with
    a JSON error, not a hang; an unknown path is a 404. Also reported, not judged: whether two identical runs give
    identical actions, whether the actions lie inside the action box, and the ``/act`` latency.
    """
    url = url.rstrip("/")
    report = ConformanceReport(url=url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(method: str, path: str, body: bytes | None = None, content_type: str = "application/json"):
        req = urllib.request.Request(url + path, data=body, method=method, headers={"Content-Type": content_type})
        t0 = time.perf_counter()
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read().decode(errors="replace")
                code = response.status
        except urllib.error.HTTPError as err:
            raw, code = err.read().decode(errors="replace"), err.code
        dt = time.perf_counter() - t0
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        return code, payload, raw, dt

    # 1. metadata
    try:
        code, meta, raw, _ = call("GET", "/metadata")
    except (urllib.error.URLError, ConnectionError, http.client.HTTPException, TimeoutError) as err:
        report.rows.append(ConformanceRow("metadata", False, f"{url}/metadata unreachable: {err}"))
        return report
    if code != 200 or not isinstance(meta, dict):
        report.rows.append(ConformanceRow("metadata", False, f"GET /metadata answered {code}: {raw[:120]!r}"))
        return report
    report.metadata = meta
    problems = []
    if meta.get("protocol") != PROTOCOL:
        problems.append(f"protocol {meta.get('protocol')!r} (expected {PROTOCOL!r})")
    if not meta.get("name"):
        problems.append("no 'name'")
    extra = sorted(k for k in meta if k not in ("protocol", "name"))
    report.rows.append(
        ConformanceRow(
            "metadata",
            not problems,
            "; ".join(problems) or f"name {meta.get('name')!r}, also reports {extra or 'nothing else'}",
        )
    )
    if problems:
        return report

    # 2. reset
    ctx = synthetic_context(num_envs=num_envs, robot_cfg=robot_cfg, action_dim=action_dim)
    instances = [
        TaskInstance(index=i, seed=i, prompt=ctx.instruction, prompt_variant="canonical") for i in range(ctx.num_envs)
    ]
    payload = {
        "context": _context_to_json(ctx),
        "env_ids": list(range(ctx.num_envs)),
        "instances": [_instance_to_json(i) for i in instances],
    }
    code, reply, raw, dt = call("POST", "/reset", json.dumps(payload).encode())
    ok = code == 200 and isinstance(reply, dict)
    report.rows.append(
        ConformanceRow(
            "reset",
            ok,
            f"{ctx.num_envs} envs, {ctx.action_dim} joints: {code} in {dt * 1000:.0f} ms"
            if ok
            else f"{code}: {raw[:160]!r}",
        )
    )
    if not ok:
        return report

    # 3. act
    def act(request: int):
        arrays = synthetic_observation(ctx)
        arrays["prompt"] = np.asarray([ctx.instruction or ""] * ctx.num_envs, dtype=str)
        arrays["request"] = np.asarray(request)
        arrays["task"] = np.asarray(ctx.task, dtype=str)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return call("POST", "/act", buf.getvalue(), content_type="application/octet-stream")

    code, reply, raw, dt = act(0)
    if code != 200 or not isinstance(reply, dict) or "actions" not in reply:
        report.rows.append(ConformanceRow("act", False, f"{code}: {raw[:160]!r}"))
        return report
    try:
        actions = np.asarray(reply["actions"], dtype=np.float32)
    except (TypeError, ValueError) as err:
        report.rows.append(ConformanceRow("act", False, f"'actions' is not a numeric array: {err}"))
        return report
    shape_ok = (actions.ndim == 2 and actions.shape == (ctx.num_envs, ctx.action_dim)) or (
        actions.ndim == 3 and actions.shape[0] == ctx.num_envs and actions.shape[2] == ctx.action_dim
    )
    finite = bool(np.isfinite(actions).all()) if actions.size else False
    horizon = actions.shape[1] if actions.ndim == 3 else 1
    report.rows.append(
        ConformanceRow(
            "act",
            shape_ok and finite,
            (
                f"shape {tuple(actions.shape)}"
                + (f" (action chunk, horizon {horizon})" if actions.ndim == 3 else "")
                + ("" if finite else "; non-finite values")
            )
            if shape_ok
            else f"shape {tuple(actions.shape)}; expected ({ctx.num_envs}, {ctx.action_dim}) or ({ctx.num_envs}, horizon, {ctx.action_dim})",
        )
    )
    if not (shape_ok and finite):
        return report
    flat = actions.reshape(-1, ctx.action_dim)
    low, high = ctx.action_low.numpy(), ctx.action_high.numpy()
    inside = bool(np.all(flat >= low - 1e-6) and np.all(flat <= high + 1e-6))
    report.rows.append(
        ConformanceRow(
            "action box",
            True,
            "every action inside [action_low, action_high]"
            if inside
            else f"actions leave the box by up to {float(np.max(np.maximum(low - flat, flat - high))):.4f} (the evaluator clips them; reported, not judged)",
        )
    )

    # 4. latency and determinism: reset again, act twice
    latencies = [dt]
    code2, _, _, _ = call("POST", "/reset", json.dumps(payload).encode())
    code3, reply2, _, dt2 = act(1)
    latencies.append(dt2)
    same = (
        code2 == 200
        and code3 == 200
        and isinstance(reply2, dict)
        and np.array_equal(np.asarray(reply2.get("actions"), dtype=np.float32), actions)
    )
    report.rows.append(
        ConformanceRow(
            "determinism",
            True,
            "identical actions after an identical reset (deterministic policy)"
            if same
            else "different actions after an identical reset (stochastic or stateful policy; reported, not judged)",
        )
    )
    report.rows.append(
        ConformanceRow("latency", True, f"/act median {np.median(latencies) * 1000:.0f} ms over {len(latencies)} calls")
    )

    # 5. partial reset: one env restarts, act still answers for all
    partial = {**payload, "env_ids": [0], "instances": [_instance_to_json(instances[0])]}
    code, reply, raw, _ = call("POST", "/reset", json.dumps(partial).encode())
    code_a, reply_a, raw_a, _ = act(2)
    shape = (
        np.asarray(reply_a.get("actions"), dtype=np.float32).shape
        if code_a == 200 and isinstance(reply_a, dict) and "actions" in reply_a
        else None
    )
    ok = code == 200 and shape is not None and shape[0] == ctx.num_envs
    report.rows.append(
        ConformanceRow(
            "partial reset",
            ok,
            f"reset of env 0 only, then act answers {shape}"
            if ok
            else f"reset {code} / act {code_a}: {(raw_a or raw)[:120]!r}",
        )
    )

    # 6. malformed act body and unknown path are refused, not hung
    code, reply, raw, dt = call("POST", "/act", b"not an npz", content_type="application/octet-stream")
    ok = code >= 400 and isinstance(reply, dict) and "error" in reply
    report.rows.append(
        ConformanceRow(
            "malformed act", ok, f"{code} with a JSON error in {dt * 1000:.0f} ms" if ok else f"{code}: {raw[:120]!r}"
        )
    )
    code, reply, raw, _ = call("GET", "/nope")
    ok = code == 404
    report.rows.append(ConformanceRow("unknown path", ok, f"GET /nope -> {code}" + ("" if ok else f": {raw[:100]!r}")))
    return report
