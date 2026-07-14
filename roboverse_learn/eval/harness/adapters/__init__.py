"""Policy adapters for the unified harness.

- ``scripted``: dependency-free baseline policies (zero / hold-pose / random) — useful
  as smoke tests, sanity baselines, and negotiation examples.
- ``template``: a copy-me reference showing the ~30-line shape of a real policy adapter.

Real model adapters (il runners, VLA policies) wrap their existing inference behind the
same :class:`~..policy.Policy` protocol; they live here so policy dependencies stay out
of the harness core.
"""

from __future__ import annotations

from .scripted import HoldPosePolicy, RandomPolicy, ZeroActionPolicy

__all__ = ["HoldPosePolicy", "RandomPolicy", "ZeroActionPolicy"]
