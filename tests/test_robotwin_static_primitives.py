"""Regression: visual-only primitives (create_visual_box) must survive into the bridge.

``create_visual_box`` (e.g. stamp_seal's stamp-target marker) builds a render-only
SAPIEN actor with no physx rigid body. Such actors do NOT appear in
``scene.get_all_actors()``, so they never get a tracked pose trajectory and were
silently dropped from the bridge -- the RoboVerse replay showed an empty table where
the native render shows a colored target square ("missing object"). The fix captures
the primitive's shape AND its static placement in the mesh hook, then
``_emit_static_primitives`` re-injects it as a fixed-pose object (shape in
``object_meshes`` + a constant-pose trajectory in ``object_traj``). These lock that
behavior with synthetic inputs -- no RoboTwin checkout or sim required.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CB = os.path.join(_HERE, "..", "tools", "robotwin_integration", "collect_bridge.py")


def _load_collect_bridge():
    spec = importlib.util.spec_from_file_location("rt_collect_bridge", _CB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_visual_only_primitive_is_emitted_as_static_object():
    cb = _load_collect_bridge()
    # Mesh hook captured a visual-only box ("box") with a static pose, plus a normal
    # tracked mesh actor ("100_seal") that already made it into object_meshes/traj.
    mesh_sink = {
        "100_seal": {"visual": "assets/objects/100_seal/visual/x.glb", "type": "mesh"},
        "box": {
            "type": "box",
            "half_size": [0.035, 0.035, 0.0005],
            "color": [0.8, 0.7, 0.2],
            "is_static": True,
            "_static_pose": {"pos": [0.1, -0.2, 0.74], "rot": [1.0, 0.0, 0.0, 0.0]},
        },
    }
    # Tracked actor is already present; the visual-only "box" is not.
    object_meshes = {"100_seal": {"visual": "assets/objects/100_seal/visual/x.glb", "type": "mesh"}}
    object_traj = {"100_seal": np.zeros((5, 7))}

    cb._emit_static_primitives(mesh_sink, object_meshes, object_traj, num_frames=5)

    # The visual-only box must now be present as a static object.
    assert "box" in object_meshes, "visual_box target marker was dropped from object_meshes"
    assert object_meshes["box"]["type"] == "box"
    assert object_meshes["box"]["half_size"] == [0.035, 0.035, 0.0005]
    # The internal _static_pose scratch key must not leak into the saved mesh.
    assert "_static_pose" not in object_meshes["box"]
    # A constant-pose (T, 7) trajectory matching the captured placement.
    assert "box" in object_traj
    assert object_traj["box"].shape == (5, 7)
    np.testing.assert_array_equal(object_traj["box"][0], [0.1, -0.2, 0.74, 1.0, 0.0, 0.0, 0.0])
    # Every frame is identical (it is a static marker).
    assert np.all(object_traj["box"] == object_traj["box"][0])


def test_tracked_actors_are_left_untouched():
    cb = _load_collect_bridge()
    # An actor already emitted (has a physx body / tracked pose) must not be overwritten,
    # even if it also somehow carries a _static_pose.
    mesh_sink = {"can": {"type": "mesh", "_static_pose": {"pos": [9, 9, 9], "rot": [1, 0, 0, 0]}}}
    object_meshes = {"can": {"type": "mesh", "visual": "real.glb"}}
    orig_traj = np.arange(7 * 3).reshape(3, 7).astype(float)
    object_traj = {"can": orig_traj}

    cb._emit_static_primitives(mesh_sink, object_meshes, object_traj, num_frames=3)

    assert object_meshes["can"] == {"type": "mesh", "visual": "real.glb"}
    np.testing.assert_array_equal(object_traj["can"], orig_traj)


def test_non_static_primitives_without_pose_are_skipped():
    cb = _load_collect_bridge()
    # A primitive captured without a _static_pose (e.g. a physx sphere already tracked
    # by name elsewhere) must not be force-emitted with a bogus pose.
    mesh_sink = {"garbage": {"type": "sphere", "radius": 0.008}}
    object_meshes: dict = {}
    object_traj: dict = {}

    cb._emit_static_primitives(mesh_sink, object_meshes, object_traj, num_frames=4)

    assert "garbage" not in object_meshes
    assert "garbage" not in object_traj
