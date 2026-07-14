"""Transports for the harness — how ObsBatch->ActionBatch crosses a process boundary.

- ``InProcessTransport`` (zero-copy, same process) + ``PolicyHandle`` (base.py)
- ``WsPolicyTransport`` / ``serve_policy`` (ws.py) — remote/isolated policy, typed frames

The runner always sees a ``PolicyHandle`` (a ``Policy``), so it is transport-agnostic.
"""

from __future__ import annotations

from .base import InProcessTransport, PolicyHandle, Transport

__all__ = ["InProcessTransport", "PolicyHandle", "Transport"]
