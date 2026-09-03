"""Unit tests for the SuperDex asset baker (no engine, no GPU).

``metasim.sim.superdex._assets`` rewrites URDFs into what SuperDex can load: every collision
geometry (mesh *or* primitive) becomes a watertight hull mesh referenced by absolute path, and
``BaseObjCfg.scale`` is baked in. These tests pin that contract with a tiny synthetic URDF so a
regression in the baker fails here, in the general suite, instead of only inside a SuperDex launch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is required by the SuperDex asset baker")

from metasim.sim.superdex import _assets
from metasim.utils.xml_safe import ET


def _write_open_shell(path: str) -> None:
    """An open box (one face removed): not watertight, like most URDF collision meshes."""
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    open_shell = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    assert not open_shell.is_watertight
    open_shell.export(path)


def _write_urdf(path: str, mesh_rel: str) -> None:
    path_dir = os.path.dirname(path)
    os.makedirs(os.path.join(path_dir, "meshes"), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0"?>
<robot name="toy">
  <material name="red"><color rgba="1 0 0 1"/></material>
  <link name="base">
    <inertial><mass value="2.5"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><mesh filename="package://{mesh_rel}"/></geometry><material name="red"/></visual>
    <collision><geometry><mesh filename="package://{mesh_rel}"/></geometry></collision>
  </link>
  <link name="tip">
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="base_to_tip" type="revolute">
    <parent link="base"/><child link="tip"/><origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
""")


@pytest.mark.general
def test_bake_urdf_replaces_collisions_with_watertight_hulls(tmp_path):
    urdf = tmp_path / "toy" / "toy.urdf"
    _write_urdf(str(urdf), "meshes/open.obj")
    _write_open_shell(str(tmp_path / "toy" / "meshes" / "open.obj"))

    baked = _assets.bake_urdf(str(urdf), cache_dir=str(tmp_path / "cache"))

    assert baked.link_names == ["base", "tip"]
    assert baked.joint_names == ["base_to_tip"]
    assert baked.joint_types == {"base_to_tip": "revolute"}
    assert baked.link_masses == {"base": 2.5}
    assert os.path.isfile(baked.path) and baked.path != str(urdf)

    root = ET.parse(baked.path).getroot()
    collision_meshes = [m.get("filename") for m in root.iter("collision") for m in m.iter("mesh")]
    assert len(collision_meshes) == 2, "both the mesh collision and the <box> primitive become hull meshes"
    for path in collision_meshes:
        assert os.path.isabs(path) and os.path.isfile(path)
        hull = trimesh.load(path, force="mesh")
        assert hull.is_watertight and hull.volume > 0
    # the box primitive is no longer referenced by anything but its hull mesh
    assert not list(root.iter("box"))
    # visual mesh path is absolute now, and the material colour was resolved
    (visual_mesh,) = [m.get("filename") for v in root.iter("visual") for m in v.iter("mesh")]
    assert os.path.isabs(visual_mesh)
    assert baked.visuals["base"][0].color == (1.0, 0.0, 0.0, 1.0)
    assert np.allclose(baked.visuals["base"][0].link_from_geom[:3, 3], [0, 0, 0.1])
    # collisions carry (hull path, link_from_geom) for rigid-object assembly
    assert len(baked.collisions["tip"]) == 1
    assert np.allclose(baked.collisions["tip"][0][1][:3, 3], [0, 0, 0.05])


@pytest.mark.general
def test_bake_urdf_bakes_scale_into_hulls_and_origins(tmp_path):
    urdf = tmp_path / "toy" / "toy.urdf"
    _write_urdf(str(urdf), "meshes/open.obj")
    _write_open_shell(str(tmp_path / "toy" / "meshes" / "open.obj"))

    unscaled = _assets.bake_urdf(str(urdf), cache_dir=str(tmp_path / "cache"))
    scaled = _assets.bake_urdf(str(urdf), scale=(2.0, 2.0, 2.0), cache_dir=str(tmp_path / "cache"))

    assert scaled.path != unscaled.path, "scale is part of the cache key"
    hull_unscaled = trimesh.load(unscaled.collisions["base"][0][0], force="mesh")
    hull_scaled = trimesh.load(scaled.collisions["base"][0][0], force="mesh")
    assert np.isclose(hull_scaled.volume, 8 * hull_unscaled.volume, rtol=1e-3)
    root = ET.parse(scaled.path).getroot()
    (joint_origin,) = [j.find("origin") for j in root.iter("joint")]
    assert joint_origin.get("xyz").split() == ["0", "0", "0.4"]


@pytest.mark.general
def test_bake_urdf_is_cached_and_content_addressed(tmp_path):
    urdf = tmp_path / "toy" / "toy.urdf"
    _write_urdf(str(urdf), "meshes/open.obj")
    _write_open_shell(str(tmp_path / "toy" / "meshes" / "open.obj"))

    first = _assets.bake_urdf(str(urdf), cache_dir=str(tmp_path / "cache"))
    mtime = os.path.getmtime(first.path)
    second = _assets.bake_urdf(str(urdf), cache_dir=str(tmp_path / "cache"))
    assert second.path == first.path
    assert os.path.getmtime(second.path) == mtime, "an unchanged source must not be re-baked"


@pytest.mark.general
def test_bake_urdf_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        _assets.bake_urdf(str(tmp_path / "nope.urdf"), cache_dir=str(tmp_path / "cache"))


@pytest.mark.general
def test_primitive_trimesh_matches_cfg_dimensions():
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg

    cube = _assets.primitive_trimesh(PrimitiveCubeCfg(name="c", size=(0.1, 0.2, 0.3), color=[1, 0, 0]))
    assert np.allclose(cube.extents, [0.1, 0.2, 0.3]) and cube.is_watertight
    sphere = _assets.primitive_trimesh(PrimitiveSphereCfg(name="s", radius=0.25, color=[0, 1, 0]))
    assert np.isclose(sphere.extents.max(), 0.5, atol=1e-3) and sphere.is_watertight
    cyl = _assets.primitive_trimesh(PrimitiveCylinderCfg(name="y", radius=0.1, height=0.4, color=[0, 0, 1]))
    assert np.isclose(cyl.extents[2], 0.4) and cyl.is_watertight
    coords, conn = _assets.mesh_to_arrays(cube)
    assert coords.dtype == np.float32 and conn.dtype == np.int32 and coords.size == 3 * len(cube.vertices)


@pytest.mark.general
def test_watertight_hull_rejects_degenerate_geometry():
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError, match="degenerate"):
        _assets.watertight_hull(flat)


@pytest.mark.general
def test_backend_registry_lists_superdex():
    """Adding a backend is one SimType + one ``SIM_BACKENDS`` entry; both must exist and agree."""
    from metasim.constants import SimType
    from metasim.utils.setup_util import SIM_BACKENDS

    assert set(SIM_BACKENDS) == {st for st in SimType if st is not SimType.ISAACLAB}
    spec = SIM_BACKENDS[SimType.SUPERDEX]
    assert spec.module == "metasim.sim.superdex" and spec.cls == "SuperdexHandler" and spec.parallel
