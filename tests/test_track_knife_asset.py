"""Regression: the track_knife task must load the knife asset, not the teapot.

``PickPlaceTrackKnife``'s tracked ``"object"`` had its usd/urdf/mjcf paths copied
from ``track_ceramic_teapot`` (all_asset/ceramic_teapot/...), while the rest of
the task is knife-specific (knife grasp-state pkl, knife centre offset). So the
simulated object (a teapot) mismatched the grasp poses the reward assumes. This
pins the corrected knife asset. Backend-free: the task class is importable and
its ScenarioCfg is inspectable without a simulator.
"""

from __future__ import annotations

import pytest


@pytest.mark.general
def test_track_knife_object_uses_knife_asset():
    from roboverse_pack.tasks.pick_place.track_knife import PickPlaceTrackKnife

    obj = next(o for o in PickPlaceTrackKnife.scenario.objects if o.name == "object")

    for path in (obj.usd_path, obj.urdf_path, obj.mjcf_path):
        assert "ceramic_teapot" not in path, f"track_knife still references the teapot asset: {path}"
        assert "knife" in path, f"track_knife object path is not the knife asset: {path}"

    # Match the sibling approach_grasp_knife.py layout exactly (mjcf lives directly
    # under knife/, not knife/mjcf/).
    assert obj.usd_path == "roboverse_data/EmbodiedGenData/all_asset/knife/usd/knife.usd"
    assert obj.urdf_path == "roboverse_data/EmbodiedGenData/all_asset/knife/knife.urdf"
    assert obj.mjcf_path == "roboverse_data/EmbodiedGenData/all_asset/knife/knife.xml"
