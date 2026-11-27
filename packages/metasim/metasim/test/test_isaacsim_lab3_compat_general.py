"""General (non-GPU) tests for the Isaac Lab 2 / Isaac Lab 3 compatibility layer of the IsaacSim handler.

Isaac Lab 3 (Isaac Sim 6) changed four things the handler leans on: asset data fields are Warp ``ProxyArray``
views with a ``.torch`` property, quaternions are ordered ``(x, y, z, w)`` instead of ``(w, x, y, z)``,
``SimulationContext.has_gui`` is a property and the render modes are gone, and ``InteractiveScene`` no longer
clones environments itself. These tests pin the pure conversion helpers under both releases by toggling the
detection flag; no simulator is needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import metasim.sim.isaacsim.isaacsim as isaac


@pytest.fixture(params=[False, True], ids=["isaaclab2", "isaaclab3"])
def lab3(monkeypatch, request):
    monkeypatch.setattr(isaac, "_ISAACLAB3", request.param)
    return request.param


class _Proxy:
    """Stand-in for Isaac Lab 3's ``ProxyArray``: exposes the tensor through ``.torch``."""

    def __init__(self, tensor):
        self.torch = tensor


@pytest.mark.general
def test_torch_view_unwraps_proxy_arrays_and_passes_tensors_through():
    t = torch.arange(3.0)
    assert isaac._t(_Proxy(t)) is t
    assert isaac._t(t) is t
    assert isaac._t(None) is None


@pytest.mark.general
def test_quaternion_order_round_trips_and_is_identity_on_isaaclab2(lab3):
    wxyz = torch.tensor([[math.cos(0.3), 0.1, 0.2, math.sin(0.3)]])
    to_sim = isaac._quat_to_sim(wxyz)
    if lab3:
        assert torch.equal(to_sim, wxyz[:, [1, 2, 3, 0]])
    else:
        assert torch.equal(to_sim, wxyz)
    assert torch.equal(isaac._quat_from_sim(to_sim), wxyz)


@pytest.mark.general
def test_state13_from_sim_returns_a_fresh_wxyz_tensor(lab3):
    sim_state = torch.zeros(2, 3, 13)
    sim_quat = torch.tensor([0.1, 0.2, 0.3, 0.9])  # the simulator's own order
    sim_state[..., 3:7] = sim_quat
    sim_state[..., 0] = 5.0
    out = isaac._state13_from_sim(_Proxy(sim_state))
    expected = sim_quat[[3, 0, 1, 2]] if lab3 else sim_quat
    assert torch.allclose(out[1, 2, 3:7], expected)
    assert out[0, 0, 0] == 5.0
    out[0, 0, 0] = -1.0  # a copy: the simulator's buffer is untouched
    assert sim_state[0, 0, 0] == 5.0


@pytest.mark.general
def test_pose7_to_sim_converts_only_the_quaternion(lab3):
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]])
    pose[0, 3:7] = torch.tensor([math.cos(0.4), math.sin(0.4), 0.0, 0.0])
    out = isaac._pose7_to_sim(pose)
    assert torch.equal(out[0, :3], pose[0, :3])
    assert torch.equal(isaac._quat_from_sim(out[0, 3:7]), pose[0, 3:7])
    if not lab3:
        assert out is pose


@pytest.mark.general
def test_has_gui_accepts_method_and_property():
    assert isaac._sim_has_gui(SimpleNamespace(has_gui=lambda: True)) is True
    assert isaac._sim_has_gui(SimpleNamespace(has_gui=False)) is False


@pytest.mark.general
def test_sensor_rendering_falls_back_to_isaaclab3_flags():
    assert isaac._sim_renders_sensors(SimpleNamespace(has_rtx_sensors=lambda: True)) is True
    assert isaac._sim_renders_sensors(SimpleNamespace(is_rendering=False, has_offscreen_render=True)) is True
    assert isaac._sim_renders_sensors(SimpleNamespace(is_rendering=False, has_offscreen_render=False)) is False
    assert isaac._sim_renders_sensors(SimpleNamespace()) is False


@pytest.mark.general
def test_render_mode_is_set_on_isaaclab2_and_skipped_on_isaaclab3():
    calls = []

    class Lab2Sim:
        RenderMode = SimpleNamespace(NO_RENDERING="none", FULL_RENDERING="full")

        def set_render_mode(self, mode):
            calls.append(mode)

    isaac._sim_set_render_mode(Lab2Sim(), "FULL_RENDERING")
    assert calls == ["full"]
    isaac._sim_set_render_mode(SimpleNamespace(), "FULL_RENDERING")  # Isaac Lab 3: no render modes, no error


@pytest.mark.general
def test_clone_environments_prefers_the_isaaclab2_method():
    calls = []
    scene = SimpleNamespace(clone_environments=lambda copy_from_source: calls.append(copy_from_source))
    isaac._clone_environments(scene)
    assert calls == [False]


@pytest.mark.general
def test_env_namespace_regex_cannot_match_nested_prims():
    """``env_.*`` matched ``env_0/franka/.../cube`` on Isaac Lab 3 and spawned 352 cubes per env."""
    import re

    pattern = re.compile("/World/envs/env_[0-9]+/cube")
    assert pattern.fullmatch("/World/envs/env_0/cube")
    assert pattern.fullmatch("/World/envs/env_17/cube")
    assert not pattern.fullmatch("/World/envs/env_0/franka/panda_link0/cube")
    src = isaac.__file__
    text = open(src).read()
    assert "/World/envs/env_.*/" not in text, "the handler must address envs with the precise env_[0-9]+ namespace"


@pytest.mark.general
def test_kit_boots_without_the_telemetry_extension(monkeypatch):
    monkeypatch.delenv("METASIM_ISAAC_TELEMETRY", raising=False)
    assert isaac._kit_args_without_telemetry(None) == isaac._KIT_TELEMETRY_EXCLUSION
    merged = isaac._kit_args_without_telemetry("--/rtx/rendermode=RaytracedLighting")
    assert merged.split() == ["--/rtx/rendermode=RaytracedLighting", isaac._KIT_TELEMETRY_EXCLUSION]
    assert isaac._kit_args_without_telemetry(merged) == merged  # idempotent
    monkeypatch.setenv("METASIM_ISAAC_TELEMETRY", "1")
    assert isaac._kit_args_without_telemetry("--x") == "--x"


def test_cfg_quaternions_follow_the_simulator_order(lab3):
    """Initial poses, camera offsets and lights are cfg tuples in MetaSim's wxyz; Isaac Lab 3 reads them as xyzw."""
    identity = isaac._quat_cfg_to_sim((1.0, 0.0, 0.0, 0.0))
    assert identity == ((0.0, 0.0, 0.0, 1.0) if lab3 else (1.0, 0.0, 0.0, 0.0))
    assert isaac._quat_cfg_to_sim([0.5, 0.5, -0.5, 0.5]) == ((0.5, -0.5, 0.5, 0.5) if lab3 else (0.5, 0.5, -0.5, 0.5))
    assert isaac._quat_cfg_to_sim(torch.tensor([0.0, 1.0, 0.0, 0.0])) == (
        (1.0, 0.0, 0.0, 0.0) if lab3 else (0.0, 1.0, 0.0, 0.0)
    )
    assert isaac._quat_cfg_to_sim(None) is None


def test_every_cfg_quaternion_write_goes_through_the_helper():
    """No ``rot=`` / ``orientation=`` cfg field in the handler takes a raw MetaSim quaternion."""
    import inspect
    import re

    source = inspect.getsource(isaac)
    raw = [
        ln.strip()
        for ln in source.splitlines()
        if re.search(r"\b(rot|orientation)=", ln) and "_quat_cfg_to_sim(" not in ln and not ln.strip().startswith("#")
    ]
    assert raw == [], raw
