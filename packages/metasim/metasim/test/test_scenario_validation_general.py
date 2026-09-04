"""Invalid scenario / sim / camera values are rejected where they are written, with the field named.

Before these checks a bad value surfaced minutes later inside a backend (a MuJoCo compile error, a
zero-sized render buffer, a NaN state) with no reference to the config field that caused it.
"""

from __future__ import annotations

import math

import pytest

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

pytestmark = pytest.mark.general


def test_valid_configs_pass_unchanged():
    s = ScenarioCfg(
        num_envs=4, decimation=3, cameras=[PinholeCameraCfg(width=64, height=32)], sim_params=SimParamCfg(dt=0.005)
    )
    assert (s.num_envs, s.decimation, s.sim_params.dt) == (4, 3, 0.005)
    assert SimParamCfg().dt is None  # None keeps the backend default


@pytest.mark.parametrize(
    "kwargs", [{"num_envs": 0}, {"num_envs": -2}, {"num_envs": 2.0}, {"num_envs": True}, {"decimation": 0}]
)
def test_scenario_rejects_non_positive_counts(kwargs):
    field = next(iter(kwargs))
    with pytest.raises(ValueError, match=rf"ScenarioCfg\.{field}=.* expected an integer >= 1"):
        ScenarioCfg(**kwargs)


def test_scenario_rejects_a_single_config_where_a_list_is_expected():
    with pytest.raises(ValueError, match=r"ScenarioCfg\.cameras=.*wrap a single config"):
        ScenarioCfg(cameras=PinholeCameraCfg())


@pytest.mark.parametrize("dt", ["", "0.01", 0, -0.001, math.nan, math.inf, True])
def test_sim_params_reject_unusable_dt(dt):
    with pytest.raises(ValueError, match=r"SimParamCfg\.dt=.* expected None or a finite number > 0"):
        SimParamCfg(dt=dt)


def test_sim_params_keep_a_valid_dt_as_given_and_reject_zero_substeps():
    assert SimParamCfg(dt=1).dt == 1  # checked, not retyped (from_dict compares types)
    with pytest.raises(ValueError, match=r"SimParamCfg\.substeps=0"):
        SimParamCfg(substeps=0)


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"height": -1}, {"width": 256.0}, {"pos": (0.0, math.nan, 1.0)}, {"pos": (1.0, 2.0)}]
)
def test_camera_rejects_empty_frames_and_non_finite_poses(kwargs):
    field = next(iter(kwargs))
    with pytest.raises(ValueError, match=rf"PinholeCameraCfg\('camera0'\)\.{field}="):
        PinholeCameraCfg(**kwargs)


def test_update_revalidates_and_leaves_no_rejected_value_behind():
    s = ScenarioCfg(num_envs=4)
    with pytest.raises(ValueError, match=r"ScenarioCfg\.num_envs=0"):
        s.update(num_envs=0)
    assert s.num_envs == 4


def test_numpy_and_torch_scalars_are_accepted_as_given():
    import numpy as np
    import torch

    s = ScenarioCfg(num_envs=np.int64(4), decimation=np.int32(3), sim_params=SimParamCfg(dt=np.float32(0.002)))
    assert s.num_envs == 4 and s.decimation == 3 and float(s.sim_params.dt) == pytest.approx(0.002)
    assert SimParamCfg(dt=torch.tensor(0.002)).dt is not None
    assert PinholeCameraCfg(width=np.int64(64), pos=np.array([0.0, 0.0, 1.0])).width == 64


def test_tuples_are_accepted_and_stored_as_lists_and_lights_are_checked():
    from metasim.scenario.lights import DistantLightCfg

    s = ScenarioCfg(objects=(), cameras=(PinholeCameraCfg(),))
    assert s.objects == [] and isinstance(s.cameras, list)
    with pytest.raises(ValueError, match=r"ScenarioCfg\.lights=.*wrap a single config"):
        ScenarioCfg(lights=DistantLightCfg())


def test_camera_intrinsics_and_look_at_are_checked():
    with pytest.raises(ValueError, match=r"PinholeCameraCfg\('wrist'\)\.horizontal_aperture=0"):
        PinholeCameraCfg(name="wrist", horizontal_aperture=0)
    with pytest.raises(ValueError, match=r"PinholeCameraCfg\('wrist'\)\.focal_length=0"):
        PinholeCameraCfg(name="wrist", focal_length=0)
    with pytest.raises(ValueError, match=r"PinholeCameraCfg\('camera0'\)\.look_at="):
        PinholeCameraCfg(look_at=(0.0, math.inf, 0.0))
    with pytest.raises(ValueError, match=r"PinholeCameraCfg\('camera0'\)\.pos="):
        PinholeCameraCfg(pos=("0", "0", "1"))
