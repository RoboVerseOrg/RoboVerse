"""Regression: URDF instance resolution must match RoboTwin's create_sapien_urdf_obj.

For URDF objects (pot/cabinet/laptop/microwave/...), ``modelid`` (random per
episode) selects WHICH physical instance is loaded. If the RoboVerse bridge picks
a different instance than RoboTwin, the replay shows "the same category but the
wrong object" -- exactly the defect this guards against. RoboTwin builds the
candidate list as the model's subdirs EXCLUDING ``visual``, sorted by numeric
name, indexed by modelid. The earlier bug included ``visual`` (it sorts to index
0), shifting every instance by one (and modelid 0 resolved to a bogus
``visual/mobility.urdf`` that crashed the replay). These lock the fixed selection
with a synthetic asset tree -- no RoboTwin checkout or sim required.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CB = os.path.join(_HERE, "..", "tools", "robotwin_integration", "collect_bridge.py")


def _load_collect_bridge():
    spec = importlib.util.spec_from_file_location("rt_collect_bridge", _CB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_assets(tmp_path, modelname, instances, with_visual=True):
    base = tmp_path / "objects" / modelname
    for inst in instances:
        d = base / inst
        d.mkdir(parents=True)
        (d / "mobility.urdf").write_text("<robot/>")
    if with_visual:
        # a shared mesh dir, NOT an instance -- RoboTwin excludes it; it has no urdf.
        (base / "visual").mkdir(parents=True)
    # a stray file that must be ignored.
    (base / "points_info.json").write_text("{}")
    return str(tmp_path / "objects")


@pytest.mark.general
def test_resolves_correct_instance_excluding_visual(tmp_path):
    """modelid indexes the numeric instances in sorted order; 'visual' is skipped."""
    cb = _load_collect_bridge()
    # mirrors the real 060_kitchenpot layout (7 instances + a visual dir).
    insts = ["100015", "100023", "100038", "100051", "100057", "100058", "100060"]
    root = _make_assets(tmp_path, "060_kitchenpot", insts)

    # modelid 0 -> first sorted instance (NOT 'visual'), modelid 2 -> third.
    assert cb.resolve_urdf_instance_dir(root, "060_kitchenpot", 0).endswith("100015")
    assert cb.resolve_urdf_instance_dir(root, "060_kitchenpot", 2).endswith("100038")
    assert cb.resolve_urdf_instance_dir(root, "060_kitchenpot", 6).endswith("100060")
    # the resolved dir always contains a real mobility.urdf, never the visual dir.
    for mid in range(len(insts)):
        d = cb.resolve_urdf_instance_dir(root, "060_kitchenpot", mid)
        assert os.path.basename(d) != "visual"
        assert os.path.exists(os.path.join(d, "mobility.urdf"))


@pytest.mark.general
def test_unsorted_dir_order_is_normalized(tmp_path):
    """Resolution is by numeric sort, independent of filesystem listing order."""
    cb = _load_collect_bridge()
    insts = ["100060", "100015", "100038"]  # deliberately unsorted
    root = _make_assets(tmp_path, "obj", insts)
    assert cb.resolve_urdf_instance_dir(root, "obj", 0).endswith("100015")
    assert cb.resolve_urdf_instance_dir(root, "obj", 1).endswith("100038")
    assert cb.resolve_urdf_instance_dir(root, "obj", 2).endswith("100060")


@pytest.mark.general
def test_modelid_none_returns_base(tmp_path):
    cb = _load_collect_bridge()
    root = _make_assets(tmp_path, "obj", ["100015"])
    assert cb.resolve_urdf_instance_dir(root, "obj", None).endswith("obj")


@pytest.mark.general
def test_out_of_range_modelid_matches_by_name(tmp_path):
    """RoboTwin's fallback: if modelid >= len, match a dir whose name == modelid."""
    cb = _load_collect_bridge()
    root = _make_assets(tmp_path, "obj", ["100015", "100023"], with_visual=False)
    # modelid 100023 is out of index range (only 2 instances) -> name match.
    assert cb.resolve_urdf_instance_dir(root, "obj", 100023).endswith("100023")
