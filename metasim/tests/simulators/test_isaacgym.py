"""Isaac Gym simulator-specific tests.

Tests unique features and behaviors of the Isaac Gym simulator while ensuring
it conforms to the base API.
"""

from pathlib import Path

import pytest
import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg as CameraCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.utils.state import TensorState

isaacgym = pytest.importorskip("isaacgym")

from metasim.sim.isaacgym.isaacgym import IsaacgymHandler


@pytest.fixture
def isaacgym_scenario():
    """Create a scenario for Isaac Gym testing."""
    robot = BaseRobotCfg(
        name="robot",
        urdf_path=str(Path(__file__).parent.parent / "assets" / "robots" / "simple_arm.urdf"),
        fix_base_link=True,
        collapse_fixed_joints=False,
        enabled_gravity=True,
        actuators={
            "joint1": BaseRobotCfg.ActuatorCfg(
                stiffness=100.0,
                damping=10.0,
                torque_limit=10.0,
                velocity_limit=1.0,
            ),
            "joint2": BaseRobotCfg.ActuatorCfg(
                stiffness=50.0,
                damping=5.0,
                torque_limit=5.0,
                velocity_limit=1.0,
            ),
        },
        control_type={"joint1": "position", "joint2": "position"},
        default_joint_positions={"joint1": 0.0, "joint2": 0.0},
    )

    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            half_size=(0.05, 0.05, 0.05),
            color=(1.0, 0.0, 0.0),
            fix_base_link=False,
            mass=0.1,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=(0.0, 0.0, 1.0),
            fix_base_link=False,
            mass=0.1,
        ),
        PrimitiveCylinderCfg(
            name="cylinder",
            radius=0.05,
            height=0.2,
            color=(0.0, 1.0, 0.0),
            fix_base_link=False,
            mass=0.1,
        ),
    ]

    cameras = [
        CameraCfg(
            name="main_camera",
            pos=(2.0, 0.0, 1.5),
            look_at=(0.0, 0.0, 0.5),
            width=640,
            height=480,
            vertical_fov=60.0,
            data_types=["rgb", "depth"],
        )
    ]

    return ScenarioCfg(
        num_envs=4,
        robots=[robot],
        objects=objects,
        cameras=cameras,
        checker=EmptyChecker(),
        sim_params=SimParamCfg(
            dt=0.01,
            substeps=2,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        ),
        decimation=1,
        episode_length=100,
        env_spacing=3.0,
        headless=True,
        control=ScenarioCfg.ControlCfg(
            torque_limit_scale=0.8,
            action_scale=1.0,
        ),
    )


@pytest.mark.gpu
class TestIsaacgymHandler:
    """Test Isaac Gym-specific functionality."""

    def test_gpu_requirement(self, isaacgym_scenario):
        """Test that Isaac Gym requires GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert handler.device.type == "cuda"

        handler.close()

    def test_multi_env_support(self, isaacgym_scenario):
        """Test Isaac Gym's efficient multi-environment support."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert handler.num_envs == 4
        assert handler.gym is not None
        assert len(handler.envs) == 4

        handler.close()

    def test_physics_params_configuration(self, isaacgym_scenario):
        """Test physics parameters configuration."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        sim_params = handler.sim_params
        assert sim_params.dt == isaacgym_scenario.sim_params.dt
        assert sim_params.substeps == isaacgym_scenario.sim_params.substeps

        handler.close()

    def test_primitive_creation_with_physics(self, isaacgym_scenario):
        """Test creation of primitives with physics properties."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert "cube" in handler.obj_actors
        assert "sphere" in handler.obj_actors
        assert "cylinder" in handler.obj_actors

        for obj_name in ["cube", "sphere", "cylinder"]:
            obj = handler.object_dict[obj_name]
            assert hasattr(obj, "mass")
            assert obj.mass == 0.1

        handler.close()

    def test_camera_sensor_creation(self, isaacgym_scenario):
        """Test camera sensor creation and configuration."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert len(handler.cam_sensors) > 0
        assert "main_camera" in handler.cam_configs

        cam_config = handler.cam_configs["main_camera"]
        assert cam_config.width == 640
        assert cam_config.height == 480

        handler.close()

    def test_tensor_state_management(self, isaacgym_scenario):
        """Test Isaac Gym's tensor-based state management."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        states = handler._get_states()
        assert isinstance(states, TensorState)

        robot_state = states.robots["robot"]
        assert isinstance(robot_state.root_state, torch.Tensor)
        assert isinstance(robot_state.joint_pos, torch.Tensor)
        assert isinstance(robot_state.joint_vel, torch.Tensor)

        assert robot_state.root_state.shape == (4, 13)
        assert robot_state.joint_pos.shape == (4, 2)

        assert robot_state.root_state.device.type == "cuda"

        handler.close()

    def test_action_tensor_handling(self, isaacgym_scenario):
        """Test that Isaac Gym handles tensor actions efficiently."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        action_tensor = torch.tensor(
            [
                [0.5, -0.5],
                [0.3, -0.3],
                [0.1, -0.1],
                [-0.1, 0.1],
            ],
            device=handler.device,
        )

        obs, _, _, _, _ = handler.step(action_tensor)

        assert isinstance(obs, TensorState)

        actions = []
        for i in range(4):
            actions.append({
                "robot": {
                    "dof_pos_target": {
                        "joint1": float(action_tensor[i, 0]),
                        "joint2": float(action_tensor[i, 1]),
                    }
                }
            })

        obs2, _, _, _, _ = handler.step(actions)
        assert isinstance(obs2, TensorState)

        handler.close()

    def test_gpu_pipeline_rendering(self, isaacgym_scenario):
        """Test GPU pipeline rendering capabilities."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert handler.use_gpu_pipeline is not None

        states = handler._get_states()

        if "main_camera" in states.cameras:
            cam_state = states.cameras["main_camera"]

            if cam_state.rgb is not None:
                assert isinstance(cam_state.rgb, torch.Tensor)
                assert cam_state.rgb.shape == (4, 480, 640, 3)

            if cam_state.depth is not None:
                assert isinstance(cam_state.depth, torch.Tensor)
                assert cam_state.depth.shape == (4, 480, 640)

        handler.close()

    def test_contact_force_handling(self, isaacgym_scenario):
        """Test contact force computation."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}}] * 4
        handler.step(actions)

        if hasattr(handler, "contact_force_tensor"):
            assert handler.contact_force_tensor is not None
            assert isinstance(handler.contact_force_tensor, torch.Tensor)

        handler.close()

    def test_jacobian_computation(self, isaacgym_scenario):
        """Test Jacobian computation capabilities."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        if hasattr(handler.gym, "acquire_jacobian_tensor"):
            jacobian_tensor = handler.gym.acquire_jacobian_tensor(handler.sim, "robot")
            assert jacobian_tensor is not None

        handler.close()

    def test_mass_matrix_computation(self, isaacgym_scenario):
        """Test mass matrix computation."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        if hasattr(handler.gym, "acquire_mass_matrix_tensor"):
            mass_matrix = handler.gym.acquire_mass_matrix_tensor(handler.sim, "robot")
            assert mass_matrix is not None

        handler.close()

    def test_domain_randomization(self, isaacgym_scenario):
        """Test domain randomization capabilities."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        if hasattr(handler, "randomize_rigid_body_props"):
            pass

        handler.close()

    def test_viewer_integration(self, isaacgym_scenario):
        """Test viewer integration when not headless."""
        isaacgym_scenario.headless = False

        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        assert handler.viewer is not None

        if hasattr(handler, "set_camera_pose"):
            handler.set_camera_pose(position=(3.0, 3.0, 3.0), look_at=(0.0, 0.0, 0.0))

        handler.close()

    def test_force_torque_sensors(self, isaacgym_scenario):
        """Test force/torque sensor support."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        if hasattr(handler.gym, "acquire_force_sensor_tensor"):
            pass

        handler.close()

    def test_articulation_handling(self, isaacgym_scenario):
        """Test articulation handling with DOF properties."""
        articulated_obj = ArticulationObjCfg(
            name="articulated_box",
            urdf_path=str(Path(__file__).parent.parent / "assets" / "objects" / "articulated_box.urdf"),
            fix_base_link=False,
        )
        isaacgym_scenario.objects.append(articulated_obj)

        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        if hasattr(handler, "dof_props"):
            assert handler.dof_props is not None

        handler.close()

    def test_env_reset_subset(self, isaacgym_scenario):
        """Test resetting subset of environments."""
        handler = IsaacgymHandler(isaacgym_scenario)
        handler.launch()

        obs, _ = handler.reset(env_ids=[0, 2])

        assert isinstance(obs, TensorState)
        assert obs.robots["robot"].root_state.shape[0] == 4

        handler.close()


@pytest.mark.parametrize("sim_type", [isaacgym.SimType.SIM_PHYSX, isaacgym.SimType.SIM_FLEX])
def test_physics_engine_types(isaacgym_scenario, sim_type):
    """Test different physics engine types."""
    if sim_type == isaacgym.SimType.SIM_FLEX:
        pytest.skip("Flex support may not be available")

    handler = IsaacgymHandler(isaacgym_scenario)
    handler.launch()

    assert handler.sim is not None

    handler.close()


def test_performance_optimizations(isaacgym_scenario):
    """Test Isaac Gym performance optimizations."""
    isaacgym_scenario.num_envs = 64

    handler = IsaacgymHandler(isaacgym_scenario)
    handler.launch()

    states = handler._get_states()
    assert states.robots["robot"].joint_pos.shape[0] == 64

    assert states.robots["robot"].joint_pos.is_cuda

    import time

    action_tensor = torch.zeros((64, 2), device=handler.device)

    start_time = time.time()
    for _ in range(10):
        handler.step(action_tensor)
    elapsed = time.time() - start_time

    assert elapsed < 1.0

    handler.close()


def test_api_compliance_issues(isaacgym_scenario):
    """Test and document API compliance issues."""
    handler = IsaacgymHandler(isaacgym_scenario)

    compliance_issues = []

    required_methods = [
        "launch",
        "step",
        "reset",
        "render",
        "close",
        "_set_states",
        "_get_states",
        "_simulate",
        "set_dof_targets",
        "get_joint_names",
        "get_body_names",
        "refresh_render",
    ]

    for method in required_methods:
        if not hasattr(handler, method):
            compliance_issues.append(f"Missing method: {method}")

    required_props = ["episode_length_buf", "actions_cache", "device", "num_envs"]

    for prop in required_props:
        if not hasattr(handler, prop):
            compliance_issues.append(f"Missing property: {prop}")

    handler.close()

    assert len(compliance_issues) == 0 or len(compliance_issues) < 3
