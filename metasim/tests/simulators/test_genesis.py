"""Genesis simulator-specific tests.

Tests unique features and behaviors of the Genesis simulator while ensuring
it conforms to the base API.
"""

from pathlib import Path

import pytest
import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg as CameraCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.sim.genesis.genesis import GenesisHandler
from metasim.utils.state import TensorState

pytest.importorskip("genesis")


@pytest.fixture
def simple_scenario():
    """Create a simple scenario for testing."""
    robot = BaseRobotCfg(
        name="robot",
        mjcf_path=str(Path(__file__).parent.parent / "assets" / "robots" / "simple_arm.xml"),
        urdf_path=str(Path(__file__).parent.parent / "assets" / "robots" / "simple_arm.urdf"),
        fix_base_link=True,
        num_joints=2,
        actuators={
            "joint1": BaseActuatorCfg(stiffness=100.0, damping=10.0),
            "joint2": BaseActuatorCfg(stiffness=50.0, damping=5.0),
        },
        control_type={"joint1": "position", "joint2": "position"},
        default_joint_positions={"joint1": 0.0, "joint2": 0.0},
        gripper_open_q=[],
        gripper_close_q=[],
        curobo_ref_cfg_name="",
        curobo_tcp_rel_pos=(0.0, 0.0, 0.0),
        curobo_tcp_rel_rot=(0.0, 0.0, 0.0),
    )

    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=[0.1, 0.1, 0.1],
            color=[1.0, 0.0, 0.0],
            fix_base_link=False,
        )
    ]

    cameras = [
        CameraCfg(
            name="main_camera",
            pos=(2.0, 0.0, 1.5),
            look_at=(0.0, 0.0, 0.5),
            width=640,
            height=480,
            data_types=["rgb", "depth"],
        )
    ]

    return ScenarioCfg(
        num_envs=4,
        robots=[robot],
        objects=objects,
        cameras=cameras,
        checker=EmptyChecker(),
        sim_params=SimParamCfg(dt=0.01),
        decimation=1,
        episode_length=100,
        try_add_table=True,
        headless=True,
    )


class TestGenesisHandler:
    """Test Genesis-specific functionality."""

    def test_multi_env_support(self, simple_scenario):
        """Test that Genesis supports multiple environments."""
        handler = GenesisHandler(simple_scenario)
        handler.launch()

        assert handler.num_envs == 4

        states = handler._get_states()
        assert isinstance(states, TensorState)
        assert states.robots["robot"].joint_pos.shape[0] == 4

        handler.close()

    def test_gpu_acceleration(self, simple_scenario):
        """Test that Genesis uses GPU acceleration when available."""
        handler = GenesisHandler(simple_scenario)
        handler.launch()

        device = handler.device
        assert isinstance(device, torch.device)

        handler.close()

    def test_batch_operations(self, simple_scenario):
        """Test batch operations across multiple environments."""
        handler = GenesisHandler(simple_scenario)
        handler.launch()

        actions = []
        for i in range(4):
            actions.append({"robot": {"dof_pos_target": {"joint1": 0.5 + i * 0.1, "joint2": -0.5 - i * 0.1}}})

        obs, rewards, dones, truncated, info = handler.step(actions)

        assert isinstance(obs, TensorState)
        assert obs.robots["robot"].joint_pos.shape[0] == 4
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

        handler.close()

    def test_genesis_specific_sim_params(self, simple_scenario):
        """Test Genesis-specific simulation parameters."""
        simple_scenario.sim_params.substeps = 2
        simple_scenario.sim_params.solver_type = 1

        handler = GenesisHandler(simple_scenario)
        handler.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}} for _ in range(4)]

        obs, _, _, _, _ = handler.step(actions)
        assert isinstance(obs, TensorState)

        handler.close()

    def test_vectorized_reset(self, simple_scenario):
        """Test vectorized reset functionality."""
        handler = GenesisHandler(simple_scenario)
        handler.launch()

        env_ids = [0, 2]
        obs = handler.reset(env_ids)

        assert isinstance(obs, TensorState)
        assert obs.robots["robot"].joint_pos.shape[0] == 4

        handler.close()

    def test_parallel_collision_detection(self, simple_scenario):
        """Test parallel collision detection across environments."""
        simple_scenario.objects.extend([
            PrimitiveCylinderCfg(
                name="cylinder",
                radius=0.05,
                height=0.2,
                color=[0.0, 1.0, 0.0],
                fix_base_link=False,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                fix_base_link=False,
            ),
        ])

        handler = GenesisHandler(simple_scenario)
        handler.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}} for _ in range(4)]

        for _ in range(10):
            handler.step(actions)

        states = handler._get_states()
        assert "cube" in states.objects
        assert "cylinder" in states.objects
        assert "sphere" in states.objects

        handler.close()

    def test_deterministic_simulation(self, simple_scenario):
        """Test that Genesis simulation is deterministic."""
        handler1 = GenesisHandler(simple_scenario)
        handler1.launch()

        handler2 = GenesisHandler(simple_scenario)
        handler2.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}} for _ in range(4)]

        obs1, _, _, _, _ = handler1.step(actions)
        obs2, _, _, _, _ = handler2.step(actions)

        pos1 = obs1.robots["robot"].joint_pos
        pos2 = obs2.robots["robot"].joint_pos

        assert torch.allclose(pos1, pos2, atol=1e-6)

        handler1.close()
        handler2.close()

    def test_material_properties(self, simple_scenario):
        """Test setting material properties in Genesis."""
        simple_scenario.objects[0].mass = 0.5

        handler = GenesisHandler(simple_scenario)
        handler.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}} for _ in range(4)]

        obs, _, _, _, _ = handler.step(actions)
        assert isinstance(obs, TensorState)

        handler.close()

    def test_camera_rendering_batch(self, simple_scenario):
        """Test camera rendering across multiple environments."""
        handler = GenesisHandler(simple_scenario)
        handler.launch()

        states = handler._get_states()

        assert "main_camera" in states.cameras
        camera_state = states.cameras["main_camera"]

        if camera_state.rgb is not None:
            assert camera_state.rgb.shape[0] == 4
            assert camera_state.rgb.shape[1:] == (480, 640, 3)

        if camera_state.depth is not None:
            assert camera_state.depth.shape[0] == 4
            assert camera_state.depth.shape[1:] == (480, 640)

        handler.close()

    def test_api_compliance(self, simple_scenario):
        """Test that Genesis handler complies with base API."""
        handler = GenesisHandler(simple_scenario)

        assert hasattr(handler, "launch")
        assert hasattr(handler, "step")
        assert hasattr(handler, "reset")
        assert hasattr(handler, "close")
        assert hasattr(handler, "_get_states")
        assert hasattr(handler, "_set_states")

        assert hasattr(handler, "device")
        assert hasattr(handler, "num_envs")
        assert hasattr(handler, "decimation")

        import inspect

        sig = inspect.signature(handler.step)
        param = sig.parameters["action"]


@pytest.mark.parametrize("num_envs", [1, 2, 8, 16])
def test_different_env_counts(simple_scenario, num_envs):
    """Test Genesis with different numbers of environments."""
    simple_scenario.num_envs = num_envs

    handler = GenesisHandler(simple_scenario)
    handler.launch()

    assert handler.num_envs == num_envs

    actions = [{"robot": {"dof_pos_target": {"joint1": 0.3, "joint2": -0.3}}} for _ in range(num_envs)]

    obs, rewards, dones, truncated, info = handler.step(actions)

    assert obs.robots["robot"].joint_pos.shape[0] == num_envs
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)

    handler.close()


@pytest.mark.gpu
def test_gpu_memory_management(simple_scenario):
    """Test GPU memory management with multiple resets."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    handler = GenesisHandler(simple_scenario)
    handler.launch()

    for _ in range(10):
        handler.reset()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}} for _ in range(4)]

        for _ in range(5):
            handler.step(actions)

    handler.close()
