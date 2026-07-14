"""WebSocket transport — run a Policy in a separate process / env, typed protocol.

Client (:class:`WsPolicyTransport`) plugs into a :class:`~.base.PolicyHandle`; server
(:func:`serve_policy`) hosts any ``Policy`` behind a ws endpoint. Frames are typed
(hello/bind/reset/infer) and carry the ObsSpec/ActionSpec with the data, so the two
sides validate their contract instead of guessing at string keys. Lifecycle: hello
(exchange PolicyCard) -> bind (derived specs) -> reset/infer.

Optional deps::  pip install websockets msgpack msgpack-numpy
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable

import torch
from loguru import logger

from ..policy import Policy, PolicyCard
from . import serialize as S


def _card_to_wire(c: PolicyCard) -> dict[str, Any]:
    return {
        "name": c.name,
        "needs_obs": S.obs_spec_to_wire(c.needs_obs),
        "produces_action": S.action_spec_to_wire(c.produces_action),
        "device": c.device,
    }


def _card_from_wire(d: dict[str, Any]) -> PolicyCard:
    return PolicyCard(
        name=d["name"],
        needs_obs=S.obs_spec_from_wire(d["needs_obs"]),
        produces_action=S.action_spec_from_wire(d["produces_action"]),
        device=d.get("device", "cpu"),
    )


class WsProtocolError(RuntimeError):
    pass


class WsPolicyTransport:
    """Client transport: forwards describe/bind/reset/infer to a remote policy server."""

    def __init__(self, url: str, *, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout
        self._ws = None
        self._loop = None
        self._card: PolicyCard | None = None

    def connect(self) -> None:
        import websockets

        self._loop = asyncio.new_event_loop()
        # ping_interval=None: this is a blocking request/response client whose event loop is
        # NOT running between frames, so it cannot answer server keepalive pings; with the
        # default 20s ping the server would drop the socket during any long client-side work
        # (e.g. a >40s sim env build between connect() and the first bind()). Disable pings.
        self._ws = self._loop.run_until_complete(websockets.connect(self.url, max_size=None, ping_interval=None))
        resp = self._rt({"type": "hello", "id": uuid.uuid4().hex}, expect="card")
        self._card = _card_from_wire(resp["card"])

    def _rt(self, frame: dict[str, Any], *, expect: str | None = None) -> dict[str, Any]:
        import websockets

        if self._ws is None:
            raise WsProtocolError("transport is closed/broken (a prior error left it unusable); create a new handle")

        async def _go() -> dict[str, Any]:
            await self._ws.send(S.encode(frame))
            return S.decode(await self._ws.recv())

        try:
            resp = self._loop.run_until_complete(asyncio.wait_for(_go(), self.timeout))
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
            # On timeout the send already fired and the reply is in flight; reusing the socket
            # would read that stale reply and desync. Close it and mark the transport broken so
            # any further call fails cleanly (the caller must build a new handle). Closing —
            # rather than just dropping the reference — is what stops a WsProtocolError from
            # leaking an open TCP connection and a pending recv on the server.
            self._abort()
            if isinstance(e, asyncio.TimeoutError):
                raise WsProtocolError(f"remote policy timed out after {self.timeout}s on {frame.get('type')!r}") from e
            raise WsProtocolError(f"connection to remote policy closed during {frame.get('type')!r}: {e}") from e
        if resp.get("type") == "error":
            raise WsProtocolError(resp.get("message", "remote policy error"))
        if expect is not None and resp.get("type") != expect:
            raise WsProtocolError(f"expected {expect!r} reply, got {resp.get('type')!r}")
        return resp

    def _abort(self) -> None:
        """Close a broken socket and mark the transport unusable (never leak the connection)."""
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            self._loop.run_until_complete(ws.close())
        except Exception as e:  # the socket is already broken; report, don't mask the original error
            logger.debug(f"[harness] closing a broken ws transport raised (ignored): {e}")

    def describe(self) -> PolicyCard:
        assert self._card is not None, "connect() first"
        return self._card

    def bind(self, obs_spec, action_spec) -> None:
        self._rt(
            {
                "type": "bind",
                "obs_spec": S.obs_spec_to_wire(obs_spec),
                "action_spec": S.action_spec_to_wire(action_spec),
            },
            expect="ok",
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._rt({"type": "reset", "env_ids": env_ids.detach().cpu().numpy()}, expect="ok")

    def infer(self, obs):
        resp = self._rt({"type": "infer", "obs": S.obs_to_wire(obs)}, expect="action")
        return S.action_from_wire(resp["action"])

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._loop.run_until_complete(self._ws.close())
            finally:
                self._ws = None
        if self._loop is not None:
            self._loop.close()  # don't leak the event loop / orphan its tasks
            self._loop = None


def _handle(policy: Policy, frame: dict[str, Any]) -> dict[str, Any]:
    t = frame.get("type")
    if t == "hello":
        return {"type": "card", "card": _card_to_wire(policy.describe())}
    if t == "bind":
        policy.bind(S.obs_spec_from_wire(frame["obs_spec"]), S.action_spec_from_wire(frame["action_spec"]))
        return {"type": "ok"}
    if t == "reset":
        policy.reset(S._to_tensor(frame["env_ids"]))  # _to_tensor copies read-only msgpack arrays
        return {"type": "ok"}
    if t == "infer":
        action = policy.act(S.obs_from_wire(frame["obs"]))
        return {"type": "action", "action": S.action_to_wire(action)}
    return {"type": "error", "message": f"unknown frame type {t!r}"}


def serve_policy(
    policy: Policy | None = None,
    *,
    factory: Callable[[], Policy] | None = None,
    host: str = "localhost",
    port: int = 8765,
    ready=None,
) -> None:
    """Host a ``Policy`` behind a ws endpoint (blocking).

    Exactly one of:

    - ``policy=<instance>`` — ONE shared instance. ``bind()``/``reset()`` mutate per-connection
      state (the derived spec, per-env history), so a second concurrent client would silently
      re-bind the first client's policy and get actions shaped for the other sim's embodiment.
      A second connection is therefore **rejected** with an error frame and closed.
    - ``factory=<callable>`` — a fresh instance per connection (the isolated multi-client mode;
      use this to serve several sim processes from one server).

    ``ready`` (optional) is called with the bound port once the server is listening.
    """
    import websockets

    if (policy is None) == (factory is None):
        raise ValueError("serve_policy needs exactly one of policy=<instance> or factory=<callable>")
    live: set[int] = set()

    async def _handler(ws):
        if factory is None:
            if live:
                await ws.send(
                    S.encode({
                        "type": "error",
                        "message": (
                            "this server hosts a single shared policy instance and already has a client; "
                            "a second connection would re-bind that instance and desync both. Start another "
                            "server, or serve with factory=<callable> for a per-connection policy."
                        ),
                    })
                )
                await ws.close()
                return
            conn_policy = policy
        else:
            conn_policy = factory()
        live.add(id(ws))
        try:
            async for raw in ws:
                try:
                    resp = _handle(conn_policy, S.decode(raw))
                except Exception as e:  # surface as a typed error frame, don't drop the connection
                    resp = {"type": "error", "message": f"{type(e).__name__}: {e}"}
                await ws.send(S.encode(resp))
        finally:
            live.discard(id(ws))
            closer = getattr(conn_policy, "close", None)
            if factory is not None and callable(closer):
                closer()  # a per-connection instance owns its resources

    async def _main():
        # ping_interval=None: the client's event loop is idle between frames and cannot answer
        # pings, so server keepalive would wrongly drop it during long client-side sim work.
        async with websockets.serve(_handler, host, port, max_size=None, ping_interval=None) as server:
            if ready is not None:
                ready(server.sockets[0].getsockname()[1])
            await asyncio.Future()  # run forever

    asyncio.run(_main())
