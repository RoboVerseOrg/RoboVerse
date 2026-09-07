"""End to end on the MuJoCo handler: the opt-in light rig replaces the asset light and changes the frame, and
the camera state carries where the camera is and how it projects."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

pytestmark = pytest.mark.mujoco


def _renderer_available() -> bool:
    import mujoco

    try:
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 8, 8)
        r.close()
        return True
    except Exception:
        return False


def _frame(lights, use_rig: bool):
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler

    scenario = ScenarioCfg(
        robots=[FrankaCfg()],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[0.8, 0.1, 0.1],
                default_position=[0.3, -0.2, 0.05],
                physics=PhysicStateType.RIGIDBODY,
            )
        ],
        cameras=[PinholeCameraCfg(name="cam", width=96, height=96, pos=(1.5, -1.5, 1.2), look_at=(0.2, -0.2, 0.2))],
        lights=lights,
        sim_params=SimParamCfg(mujoco_use_scenario_lights=use_rig),
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    h = MujocoHandler(scenario)
    h.launch()
    try:
        h.simulate()
        rgb = h.get_states(mode="tensor").cameras["cam"].rgb[0].cpu().numpy().astype(np.float32)
        return rgb, int(h._mj_model.nlight), [int(a) for a in h._mj_model.light_active]
    finally:
        h.close()


def test_mujoco_scenario_light_rig_is_opt_in_and_reaches_the_frame():
    if not _renderer_available():
        pytest.skip("no MuJoCo offscreen renderer in this environment")
    rig = [DistantLightCfg(name="sun", intensity=2000.0, polar=30.0, azimuth=45.0)]
    default_rgb, n_default, active_default = _frame(rig, use_rig=False)
    assert sum(active_default) == n_default  # flag off: asset lights untouched, rig not added
    rig_rgb, n_rig, active_rig = _frame(rig, use_rig=True)
    assert n_rig == n_default + 1 and sum(active_rig) == 1, (n_default, active_default, n_rig, active_rig)
    assert np.abs(rig_rgb - default_rgb).mean() > 5.0, "the declared rig did not change the frame"


@pytest.mark.skipif(not _renderer_available(), reason="no offscreen GL context")
def test_camera_state_carries_the_physics_pose_and_intrinsics():
    """A world camera reports its configured pose; a camera mounted on the hand reports where the hand is and
    follows it when a joint moves (the pose comes from the physics camera, not the config)."""
    import torch

    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler
    from metasim.utils.math import quat_apply

    world = PinholeCameraCfg(name="world", width=32, height=24, pos=(1.5, -1.5, 1.2), look_at=(0.2, -0.2, 0.2))
    wrist = PinholeCameraCfg(
        name="wrist",
        width=32,
        height=24,
        mount_to="franka",
        mount_link="panda_hand",
        mount_pos=(0.0, 0.0, 0.1),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
    )
    scenario = ScenarioCfg(robots=[FrankaCfg()], cameras=[world, wrist], simulator="mujoco", num_envs=1, headless=True)
    h = MujocoHandler(scenario)
    h.launch()
    try:
        states = h.get_states(mode="tensor")
        cam = states.cameras["world"]
        assert torch.allclose(cam.pos, torch.tensor([[1.5, -1.5, 1.2]]), atol=1e-5)
        forward = torch.tensor(world.look_at) - torch.tensor(world.pos)
        forward = forward / forward.norm()
        assert torch.allclose(quat_apply(cam.quat_world, torch.tensor([[1.0, 0.0, 0.0]]))[0], forward, atol=1e-4), (
            "+X rotated by quat_world is the viewing direction"
        )
        assert quat_apply(cam.quat_world, torch.tensor([[0.0, 0.0, 1.0]]))[0, 2] > 0.5, "the camera is upright"
        assert torch.allclose(cam.intrinsics, torch.tensor([world.intrinsics]))

        before = states.cameras["wrist"]
        assert not torch.allclose(before.pos[0], torch.tensor(wrist.pos)), (
            "a mounted camera is on the hand, not at cfg.pos"
        )
        joint_names = h.get_joint_names("franka", sort=True)
        dof = dict(zip(joint_names, states.robots["franka"].joint_pos[0].tolist(), strict=True))
        dof["panda_joint1"] += 1.0  # the base joint: the hand and its camera swing about world +Z
        root = states.robots["franka"].root_state[0]
        h.set_states([{"objects": {}, "robots": {"franka": {"pos": root[:3], "rot": root[3:7], "dof_pos": dof}}}])
        after = h.get_states(mode="tensor").cameras["wrist"]
    finally:
        h.close()
    # turning the base joint by 1 rad rotates the hand, and the camera on it, by 1 rad about the base's +Z
    from metasim.utils.math import matrix_from_quat

    c, s_ = float(torch.cos(torch.tensor(1.0))), float(torch.sin(torch.tensor(1.0)))
    rz = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    base = root[:3]
    assert torch.allclose(after.pos[0], base + rz @ (before.pos[0] - base), atol=2e-3), "the camera moved with the hand"
    assert torch.allclose(
        matrix_from_quat(after.quat_world)[0], rz @ matrix_from_quat(before.quat_world)[0], atol=2e-3
    ), "and turned with it"
