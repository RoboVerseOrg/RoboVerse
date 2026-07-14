"""CLI: serve a scripted policy over WebSocket — the isolation entry point.

Run a policy in its OWN process/env, then evaluate a task against it from the sim process:

    python -m roboverse_learn.eval.harness.transport.serve --policy zero --port 8799
    # elsewhere: evaluate(task, PolicyHandle(WsPolicyTransport("ws://localhost:8799")), ...)

A real model adapter is served the same way; only its ``install.sh``/env differ.
"""

from __future__ import annotations

import argparse

from ..adapters import HoldPosePolicy, RandomPolicy, ZeroActionPolicy
from .ws import serve_policy

_POLICIES = {"zero": ZeroActionPolicy, "hold_pose": HoldPosePolicy, "random": RandomPolicy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="zero", choices=sorted(_POLICIES))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"[serve] {args.policy} policy on ws://{args.host}:{args.port}", flush=True)
    # factory=: one policy instance per connection, so two sim processes can share this server
    # without re-binding each other's spec.
    serve_policy(factory=_POLICIES[args.policy], host=args.host, port=args.port)


if __name__ == "__main__":
    main()
