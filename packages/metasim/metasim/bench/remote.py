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
                    return json.loads(response.read().decode())
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
        actions = np.asarray(reply["actions"], dtype=np.float32)
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
                            reply = server.handle_reset(json.loads(body.decode()))
                        else:
                            reply = server.handle_act(body)
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

    def handle_act(self, body: bytes) -> dict[str, Any]:
        """``POST /act``: decode the observation, call ``policy.act``, return the actions as JSON."""
        if self.ctx is None:
            raise RuntimeError("act before reset: the client must POST /reset first")
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
