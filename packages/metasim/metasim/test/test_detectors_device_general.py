"""General (non-GPU) tests: detector constants live on the simulator's device.

``Relative2DSphereDetector.reset`` built its offset on the CPU and added it to a position read from the
handler; on a CUDA backend (Newton, Isaac Sim) that raised ``Expected all tensors to be on the same device``
for every task with a goal region (ManiSkill push_cube / pull_cube). The ``meta`` device plays the role of
the GPU here: mixing it with a CPU tensor raises the same error, and no GPU is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from metasim.task.checkers import detectors


@pytest.mark.general
def test_relative_2d_sphere_detector_offset_follows_the_base_position_device(monkeypatch):
    base = torch.zeros(3, 3, device="meta")
    monkeypatch.setattr(detectors, "get_pos", lambda handler, name, env_ids=None: base)
    det = detectors.Relative2DSphereDetector(
        base_obj_name="goal", relative_pos=(0.1, 0.0, 0.0), axis=(0, 1), radius=0.1
    )
    det.reset(SimpleNamespace(num_envs=3, device=torch.device("meta")))
    assert det.checker_pos.device.type == "meta"
    assert det.checker_pos.shape == (3, 3)
