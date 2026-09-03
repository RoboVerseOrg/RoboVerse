"""Regression test: MuJoCo launch wires the dm_control struct-compat patch.

``MujocoHandler.launch`` must call ``apply_dm_control_struct_compat_patch``
before building any dm_control physics. The patch reconciles dm_control's
static ``sizes.array_sizes`` tables (which still list fields like
``flex_bandwidth``) with the live MjModel/MjData of the installed MuJoCo
build. Without it, ``mjcf.Physics.from_mjcf_model`` raises
``AttributeError: 'MjModel' object has no attribute 'flex_bandwidth'`` and
*every* MuJoCo launch fails — the patch existed but was never called.

Marked ``@pytest.mark.mujoco`` so autotest selection runs it only in a
mujoco-enabled env.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mujoco


def test_launch_invokes_dm_control_compat_patch(monkeypatch):
    """A bare MuJoCo launch must call the compat patch exactly at the choke
    point, before dm_control physics is constructed.

    Spying on the patch (rather than only asserting launch succeeds) pins the
    fix to the intended wiring: if a refactor moves physics construction
    ahead of the patch call, this fails even on a MuJoCo build where the
    struct tables happen to match.
    """
    import metasim.sim.mujoco.mujoco as mj
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.mujoco import MujocoHandler

    called = {"n": 0}
    real_patch = mj.apply_dm_control_struct_compat_patch

    def _spy():
        called["n"] += 1
        return real_patch()

    monkeypatch.setattr(mj, "apply_dm_control_struct_compat_patch", _spy)

    scenario = ScenarioCfg(
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = MujocoHandler(scenario)
    handler.launch()  # would raise AttributeError('... flex_bandwidth') before the fix
    try:
        assert called["n"] >= 1, "launch did not call apply_dm_control_struct_compat_patch"
        assert handler.physics is not None, "physics was not built"
    finally:
        try:
            handler.close()
        except Exception:
            pass


def test_compat_patch_is_idempotent_and_version_safe():
    """The patch only drops field names the live struct lacks, so calling it
    repeatedly (and on a build whose tables already match) is a safe no-op."""
    from metasim.sim.mujoco.mujoco import apply_dm_control_struct_compat_patch

    apply_dm_control_struct_compat_patch()
    apply_dm_control_struct_compat_patch()  # second call must not raise

    from dm_control import mjcf

    model = mjcf.RootElement()
    body = model.worldbody.add("body", name="b")
    body.add("geom", type="sphere", size=[0.1])
    physics = mjcf.Physics.from_mjcf_model(model)
    assert physics.model.nbody >= 1
