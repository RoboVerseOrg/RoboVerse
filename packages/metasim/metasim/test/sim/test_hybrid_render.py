"""Hybrid simulation: MuJoCo steps the physics, Isaac Sim renders the frames.

``HybridSimHandler`` pushes the physics state into the render handler after every step, so the
rendered scene must show exactly the poses MuJoCo computed, and the camera outputs must have the
shapes the ``CameraState`` contract promises. This is the "many-env rendering" path: N MuJoCo
worker processes for physics, one Isaac Sim stage with N envs for RTX rendering.

Runs only where Isaac Sim + Isaac Lab are installed; the module is skipped elsewhere. The ``hybrid``
fixture is parametrized with ``{"sim": "isaacsim"}`` so ``pytest -k isaacsim`` selects these tests and the
session fixture in ``metasim/test/conftest.py`` starts the Isaac Sim app for them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from loguru import logger as log

pytest.importorskip("isaacsim")
pytest.importorskip("isaaclab")
pytest.importorskip("mujoco")

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.hybrid import HybridSimHandler
from metasim.utils.setup_util import get_sim_handler_class

pytestmark = pytest.mark.sim("isaacsim")


def _scenario(sim: str, num_envs: int) -> ScenarioCfg:
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg

    return ScenarioCfg(
        robots=[FrankaCfg()],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                default_position=[0.3, -0.2, 0.05],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                default_position=[0.4, -0.6, 0.1],
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        cameras=[PinholeCameraCfg(name="cam", width=256, height=256, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))],
        simulator=sim,
        num_envs=num_envs,
        headless=True,
    )


def _make_hybrid(num_envs: int, simulation_app):
    # physics never renders in hybrid mode: no cameras, so MuJoCo does not open a GL context
    physics = get_sim_handler_class(SimType.MUJOCO)(_scenario("mujoco", num_envs).replace(cameras=[]))
    renderer = get_sim_handler_class(SimType.ISAACSIM)(_scenario("isaacsim", num_envs))
    hybrid = HybridSimHandler(_scenario("mujoco", num_envs), physics, renderer)
    # reuse the session's Kit instance: a second AppLauncher in one process shuts the first down
    hybrid.launch(simulation_app=simulation_app)
    return hybrid


NUM_ENVS = 4


@pytest.fixture(scope="module", params=[{"sim": "isaacsim"}], ids=["isaacsim"])
def hybrid(request, isaacsim_app):
    """One N-env hybrid for the whole module.

    Isaac Sim keeps one stage per process, so a second handler in the same Kit instance collides
    with the first one's prims.
    """
    h = _make_hybrid(NUM_ENVS, isaacsim_app)
    yield h
    h.close()


def _drive(hybrid, steps: int) -> float:
    """Drive every env to a different joint-1 target for ``steps`` steps; returns seconds per step."""
    names = hybrid.physics_handler.get_joint_names("franka", sort=True)
    target = torch.zeros(NUM_ENVS, len(names))
    target[:, names.index("panda_joint1")] = torch.linspace(-0.5, 0.5, NUM_ENVS)
    t0 = time.perf_counter()
    for _ in range(steps):
        hybrid.set_dof_targets(target)
        hybrid.simulate()
    return (time.perf_counter() - t0) / steps


def test_render_state_follows_physics_state(hybrid):
    """After a step, the renderer's poses equal MuJoCo's poses in every env (the sync is the point)."""
    _drive(hybrid, 50)
    phys = hybrid.physics_handler.get_states(mode="tensor")
    rend = hybrid.render_handler.get_states(mode="tensor")
    for name in ("cube", "sphere"):
        p, r = phys.objects[name].root_state[:, :7].cpu(), rend.objects[name].root_state[:, :7].cpu()
        assert torch.allclose(p, r, atol=1e-5), f"{name}: physics {p.tolist()} vs render {r.tolist()}"
    pq, rq = phys.robots["franka"].joint_pos.cpu(), rend.robots["franka"].joint_pos.cpu()
    assert torch.allclose(pq, rq, atol=1e-5), f"franka joints: physics {pq.tolist()} vs render {rq.tolist()}"
    # the envs were driven to different targets, so the physics side must differ across envs
    assert not torch.allclose(pq[0], pq[-1])


def test_camera_outputs_have_contract_shapes(hybrid):
    """RGB is (N, H, W, 3) uint8, depth is (N, H, W); both come from the render handler."""
    state = hybrid.get_states(mode="tensor")
    cam = state.cameras["cam"]
    assert cam.rgb is not None and tuple(cam.rgb.shape) == (NUM_ENVS, 256, 256, 3) and cam.rgb.dtype == torch.uint8
    assert cam.depth is not None and tuple(cam.depth.shape) == (NUM_ENVS, 256, 256)


def test_many_env_rendering(hybrid):
    """N physics workers + one N-env Isaac Sim stage: every env renders its own scene.

    The per-step cost is logged so regressions in the state sync show up.
    """
    dt = _drive(hybrid, 20)
    frames = hybrid.get_states(mode="tensor").cameras["cam"].rgb.cpu().numpy().astype(np.float32)
    assert all(frames[i].std() > 5 for i in range(NUM_ENVS)), "an env rendered a flat frame"
    # envs were driven to different joint targets: their frames must differ
    assert np.abs(frames[0] - frames[-1]).mean() > 0.5
    log.info(f"[hybrid] num_envs={NUM_ENVS}: {dt * 1000:.1f} ms per simulate() incl. render sync")


def test_single_render_pass_shows_the_new_state(hybrid):
    """The per-step sync renders once, and one pass must already show a teleported object.

    The frame changes versus before the teleport and matches a full two-pass refresh of the same state.
    """
    before = hybrid.get_states(mode="tensor").cameras["cam"].rgb.cpu().numpy().astype(np.float32)
    state = hybrid.get_states(mode="tensor")
    state.objects["cube"].root_state[:, 0] += 0.35  # slide the cube along +x in every env
    state.objects["cube"].root_state[:, 7:] = 0.0
    hybrid.set_states(state)  # physics write + one deferred-flush render pass
    one_pass = hybrid.get_states(mode="tensor").cameras["cam"].rgb.cpu().numpy().astype(np.float32)
    hybrid.render_handler.refresh_render(passes=2)
    hybrid.render_handler.invalidate_state_caches()
    hybrid.invalidate_state_caches()
    two_pass = hybrid.get_states(mode="tensor").cameras["cam"].rgb.cpu().numpy().astype(np.float32)
    assert np.abs(one_pass - before).mean() > 0.5, "the frame did not change after the cube moved"
    assert np.abs(one_pass - two_pass).mean() < 0.5, "one render pass lags behind the state"


def test_wrapped_handlers_see_the_hybrid_step(hybrid):
    """The wrapped handlers' own ``get_states`` reflect a hybrid step.

    ``hybrid.simulate()`` / ``hybrid.set_states()`` drive the wrapped handlers through their private
    API, so the hybrid must invalidate their public ``get_states`` caches: a user who reads
    ``hybrid.render_handler.get_states()`` after a step must see the step, not the cached frame.
    """
    rend_before = hybrid.render_handler.get_states(mode="tensor").objects["cube"].root_state[:, :3].clone()
    phys_before = hybrid.physics_handler.get_states(mode="tensor").objects["cube"].root_state[:, :3].clone()
    state = hybrid.get_states(mode="tensor")
    state.objects["cube"].root_state[:, 1] -= 0.2
    state.objects["cube"].root_state[:, 7:] = 0.0
    hybrid.set_states(state)
    phys_after = hybrid.physics_handler.get_states(mode="tensor").objects["cube"].root_state[:, :3]
    rend_after = hybrid.render_handler.get_states(mode="tensor").objects["cube"].root_state[:, :3]
    assert not torch.allclose(phys_after.cpu(), phys_before.cpu()), "physics cache was not invalidated"
    assert not torch.allclose(rend_after.cpu(), rend_before.cpu()), "renderer cache was not invalidated"
    assert torch.allclose(phys_after.cpu(), rend_after.cpu(), atol=1e-5)
    # and a plain step: the physics side moves (joints are driven), the cached copy must not survive
    q_before = hybrid.physics_handler.get_states(mode="tensor").robots["franka"].joint_pos.clone()
    _drive(hybrid, 5)
    q_after = hybrid.physics_handler.get_states(mode="tensor").robots["franka"].joint_pos
    assert not torch.equal(q_after.cpu(), q_before.cpu()), "physics cache survived hybrid.simulate()"


def test_set_states_subset_env_ids_reaches_the_renderer(hybrid):
    """A partial reset moves env k in the renderer and leaves the other envs where they were.

    ``set_states(..., env_ids=[k])``: the physics side returns subset rows while the renderer indexes
    a full batch, and ``get_states(env_ids=[k])`` must slice both sides to one env count.
    """
    k = NUM_ENVS - 1
    full = hybrid.get_states(mode="tensor")
    before = hybrid.render_handler.get_states(mode="tensor").objects["sphere"].root_state[:, :3].clone().cpu()
    sub = hybrid.get_states(env_ids=[k], mode="tensor")
    sub.objects["sphere"].root_state[:, 0] -= 0.25
    sub.objects["sphere"].root_state[:, 7:] = 0.0
    hybrid.set_states(sub, env_ids=[k])
    phys = hybrid.physics_handler.get_states(mode="tensor").objects["sphere"].root_state[:, :3].cpu()
    rend = hybrid.render_handler.get_states(mode="tensor").objects["sphere"].root_state[:, :3].cpu()
    assert torch.allclose(phys[k], rend[k], atol=1e-5), (
        f"env {k}: physics {phys[k].tolist()} vs render {rend[k].tolist()}"
    )
    assert abs(float(rend[k, 0] - before[k, 0]) + 0.25) < 1e-4, "env k did not move in the renderer"
    others = [i for i in range(NUM_ENVS) if i != k]
    assert torch.allclose(rend[others], before[others], atol=1e-5), "an untouched env moved"
    assert torch.allclose(full.objects["sphere"].root_state[others, :3].cpu(), phys[others], atol=1e-5)
