"""Backend-free regression: embodiedgen tasks use a sane max_episode_steps.

``put_banana`` carried ``max_episode_steps = 250000`` (a typo — base and the
``put_mug`` sibling both use 250), which would let an unsolved episode run for
250k steps. This statically (no sim backend) pins every embodiedgen task's
declared ``max_episode_steps`` to the base value.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_EMBODIEDGEN = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "embodiedgen"
_EXPECTED = 250  # base.py / put_mug.py convention


def _max_episode_steps(path: pathlib.Path) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "max_episode_steps" and isinstance(node.value, ast.Constant):
                    return node.value.value
    return None


@pytest.mark.general
def test_embodiedgen_max_episode_steps_are_sane():
    offenders = {}
    for path in sorted(_EMBODIEDGEN.glob("*.py")):
        val = _max_episode_steps(path)
        if val is not None and val != _EXPECTED:
            offenders[path.name] = val
    assert not offenders, f"embodiedgen tasks with off-convention max_episode_steps (expected {_EXPECTED}): {offenders}"
