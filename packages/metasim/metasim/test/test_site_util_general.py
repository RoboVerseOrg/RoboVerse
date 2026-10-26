"""``site_pose`` computes a site's world pose from the state; MuJoCo's own kinematics is the reference."""

from __future__ import annotations

import re
import types

import numpy as np
import pytest
import torch

from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.utils.site_util import site_pose

mujoco = pytest.importorskip("mujoco")

_ASSET = """
<mujoco>
  <compiler angle="{angle}"/>
  <worldbody>
    <body pos="0.1 0 0.2" euler="{euler}">
      <body name="object" pos="0 0.05 0" axisangle="{axisangle}">
        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        <site name="on_object" pos="0.01 0.02 0.07" quat="0 0 0 1"/>
        <body name="lid" pos="0 0 0.1" quat="0.9238795 0 0.3826834 0">
          <joint name="hinge" type="hinge" axis="1 0 0"/>
          <geom type="box" size="0.05 0.05 0.01" mass="0.1"/>
          <body pos="0.02 0 0.01" euler="{euler2}">
            <site name="on_lid" pos="0 0.03 0.02" euler="{euler2}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_FORMS = {
    "degree": {"euler": "10 20 30", "axisangle": "0 1 1 40", "euler2": "-15 5 25"},
    "radian": {"euler": "0.1 0.2 0.3", "axisangle": "0 1 1 0.7", "euler2": "-0.25 0.1 0.4"},
}


def _wrapped(asset: str, pos, quat) -> str:
    """The asset under a body at (pos, quat): what a simulator does when it places the object."""
    body = f'<body name="root" pos="{" ".join(map(str, pos))}" quat="{" ".join(map(str, quat))}">'
    return asset.replace("<worldbody>", "<worldbody>" + body, 1).replace("</worldbody>", "</body></worldbody>", 1)


def _state(model, data, *, name: str, body_names: list[str]):
    def pose(body: str):
        return np.concatenate([data.body(body).xpos, data.body(body).xquat])

    root = torch.tensor(np.concatenate([pose("root"), np.zeros(6)]), dtype=torch.float64)[None]
    bodies = torch.tensor(np.stack([np.concatenate([pose(b), np.zeros(6)]) for b in body_names]), dtype=torch.float64)
    obj = types.SimpleNamespace(root_state=root, body_names=body_names, body_state=bodies[None])
    return types.SimpleNamespace(objects={name: obj})


def _site_world(model, data, site: str):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site(site).xmat)
    return data.site(site).xpos.copy(), quat


@pytest.mark.parametrize("angle", ["degree", "radian"])
@pytest.mark.parametrize("articulated", [False, True])
def test_site_pose_matches_mujoco(tmp_path, angle, articulated):
    asset = _ASSET.format(angle=angle, **_FORMS[angle])
    path = tmp_path / "asset.xml"
    path.write_text(asset)
    model = mujoco.MjModel.from_xml_string(_wrapped(asset, (0.3, -0.2, 0.8), (0.8660254, 0, 0, 0.5)))
    data = mujoco.MjData(model)
    if articulated:
        data.qpos[0] = 0.6  # the lid's hinge moves the lid's site, which a root-relative offset cannot follow
    mujoco.mj_forward(model, data)

    cls = ArticulationObjCfg if articulated else RigidObjCfg
    obj = cls(name="box", mjcf_path=str(path))
    states = _state(model, data, name="box", body_names=["lid", "object"])
    for site in ("on_object", "on_lid"):
        got = site_pose(states, obj, site)
        pos, quat = _site_world(model, data, site)
        np.testing.assert_allclose(got.pos[0].numpy(), pos, atol=1e-9)
        sign = 1.0 if float(np.dot(got.quat[0].numpy(), quat)) >= 0 else -1.0
        np.testing.assert_allclose(sign * got.quat[0].numpy(), quat, atol=1e-9)
        np.testing.assert_allclose(got.rot[0].numpy(), data.site(site).xmat.reshape(3, 3), atol=1e-9)


def test_site_pose_follows_scale(tmp_path):
    """Scale multiplies every body and site ``pos`` (what the MuJoCo handler does to the MJCF), not rotations."""
    asset = _ASSET.format(angle="degree", **_FORMS["degree"])
    path = tmp_path / "asset.xml"
    path.write_text(asset)
    scale = (2.0, 0.5, 1.5)

    def scaled(match):
        x, y, z = (float(v) * s for v, s in zip(match.group(1).split(), scale))
        return f'pos="{x} {y} {z}"'

    model = mujoco.MjModel.from_xml_string(_wrapped(re.sub(r'pos="([^"]+)"', scaled, asset), (0, 0, 0), (1, 0, 0, 0)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    states = _state(model, data, name="box", body_names=["lid", "object"])
    got = site_pose(states, RigidObjCfg(name="box", mjcf_path=str(path), scale=scale), "on_lid")
    np.testing.assert_allclose(got.pos[0].numpy(), _site_world(model, data, "on_lid")[0], atol=1e-9)


def test_site_pose_names_what_is_missing(tmp_path):
    path = tmp_path / "asset.xml"
    path.write_text(_ASSET.format(angle="degree", **_FORMS["degree"]))
    state = types.SimpleNamespace(
        root_state=torch.zeros(1, 13), body_names=["object"], body_state=torch.zeros(1, 1, 13)
    )
    states = types.SimpleNamespace(objects={"box": state})
    with pytest.raises(KeyError, match="declares no site 'nope'"):
        site_pose(states, RigidObjCfg(name="box", mjcf_path=str(path)), "nope")
    with pytest.raises(KeyError, match="fixed to body 'lid'"):
        site_pose(states, ArticulationObjCfg(name="box", mjcf_path=str(path)), "on_lid")
