"""Regression: draw_svg / draw_triangle _terminated returns a per-env (num_envs,) tensor.

Both tasks returned ``torch.tensor([False])`` — a fixed shape ``(1,)`` on the
default device — regardless of num_envs, mismatching the ``(num_envs,)`` timeout
/ done bookkeeping when num_envs > 1. They now return
``torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)``. Backend-free:
``_terminated`` ignores its ``states`` argument, so it can be called on a
``__new__`` instance with num_envs/device set.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.general
@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("roboverse_pack.tasks.maniskill.draw_svg", "DrawSvgTask"),
        ("roboverse_pack.tasks.maniskill.draw_triangle", "DrawTriangleTask"),
    ],
)
def test_draw_terminated_is_per_env(module_name, class_name):
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)
    task = cls.__new__(cls)  # bypass __init__/backend
    task.num_envs = 4
    task.device = torch.device("cpu")

    out = task._terminated(None)  # _terminated ignores states

    assert out.shape == (4,), f"{class_name}._terminated must be (num_envs,), got {tuple(out.shape)}"
    assert out.dtype == torch.bool
    assert not out.any()
