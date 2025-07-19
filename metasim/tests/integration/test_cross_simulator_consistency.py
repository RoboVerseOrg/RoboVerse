"""Cross-simulator consistency tests.

These tests ensure that the same scenario produces consistent results
across different simulators, within reasonable tolerances.
"""

from pathlib import Path

import pytest
import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg as CameraCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_handler_class
from metasim.utils.state import TensorState

POSITION_TOLERANCE = 1e-2
ORIENTATION_TOLERANCE = 1e-2
VELOCITY_TOLERANCE = 5e-2
JOINT_TOLERANCE = 1e-3


@pytest.fixture
def reference_scenario():
    """Create a reference scenario for cross-simulator testing."""
    robot = BaseRobotCfg(
        name="robot",
        urdf_path=str(Path(__file__).parent.parent / "assets" / "robots" / "simple_arm.urdf"),
        mjcf_path=str(Path(__file__).parent.parent / "assets" / "robots" / "simple_arm.xml"),
        fix_base_link=True,
        collapse_fixed_joints=False,
        enabled_gravity=True,
        actuators={
            "joint1": BaseActuatorCfg(
                stiffness=100.0,
                damping=10.0,
                torque_limit=10.0,
                velocity_limit=1.0,
                fully_actuated=True,
            ),
            "joint2": BaseActuatorCfg(
                stiffness=50.0,
                damping=5.0,
                torque_limit=5.0,
                velocity_limit=1.0,
                fully_actuated=True,
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
        )
    ]

    cameras = [
        CameraCfg(
            name="main_camera",
            pos=(2.0, 0.0, 1.5),
            look_at=(0.0, 0.0, 0.5),
            width=320,
            height=240,
            vertical_fov=60.0,
            data_types=["rgb"],
        )
    ]

    return ScenarioCfg(
        num_envs=1,
        robots=[robot],
        objects=objects,
        cameras=cameras,
        checker=EmptyChecker(),
        sim_params=SimParamCfg(
            dt=0.01,
            substeps=1,
        ),
        decimation=1,
        episode_length=100,
        env_spacing=3.0,
        headless=True,
        try_add_table=True,
    )


def get_available_simulators():
    """Get list of available simulator handlers for testing."""
    available = []

    simulators_to_test = [
        (SimType.MUJOCO, lambda: pytest.importorskip("mujoco")),
        (SimType.GENESIS, lambda: pytest.importorskip("genesis")),
        (SimType.ISAACGYM, lambda: pytest.importorskip("isaacgym")),
        (SimType.ISAACLAB, lambda: pytest.importorskip("omni.isaac.lab")),
        (SimType.SAPIEN2, lambda: pytest.importorskip("sapien")),
        (SimType.PYBULLET, lambda: pytest.importorskip("pybullet")),
    ]

    for sim_type, import_func in simulators_to_test:
        try:
            import_func()
            handler_class = get_sim_handler_class(sim_type)
            if handler_class is not None:
                available.append(sim_type)
        except:
            pass

    return available


@pytest.mark.integration
class TestCrossSimulatorConsistency:
    """Test consistency across different simulators."""

    @pytest.fixture(scope="class")
    def simulator_results(self, reference_scenario):
        """Run the same scenario on all available simulators and collect results."""
        results = {}
        available_sims = get_available_simulators()

        if len(available_sims) < 2:
            pytest.skip("Need at least 2 simulators for cross-validation")

        for sim_type in available_sims:
            try:
                handler_class = get_sim_handler_class(sim_type)
                handler = handler_class(reference_scenario)
                handler.launch()

                initial_state = handler._get_states()

                action_sequence = [
                    {"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}},
                    {"robot": {"dof_pos_target": {"joint1": -0.5, "joint2": 0.5}}},
                    {"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}},
                ]

                states_over_time = []
                for action in action_sequence:
                    for _ in range(10):
                        obs, _, _, _, _ = handler.step([action])
                        states_over_time.append(obs)

                final_state = handler._get_states()

                results[sim_type] = {
                    "initial_state": initial_state,
                    "final_state": final_state,
                    "states_over_time": states_over_time,
                    "handler": handler,
                }

                handler.close()

            except Exception as e:
                print(f"Failed to test {sim_type}: {e}")
                continue

        return results

    def test_initial_state_consistency(self, simulator_results):
        """Test that initial states are consistent across simulators."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        sim_types = list(simulator_results.keys())
        reference_sim = sim_types[0]
        reference_state = simulator_results[reference_sim]["initial_state"]

        for sim_type in sim_types[1:]:
            compare_state = simulator_results[sim_type]["initial_state"]

            ref_robot_pos = reference_state.robots["robot"].root_state[0, :3]
            comp_robot_pos = compare_state.robots["robot"].root_state[0, :3]

            pos_diff = torch.norm(ref_robot_pos - comp_robot_pos).item()
            assert pos_diff < POSITION_TOLERANCE, (
                f"{sim_type} robot position differs from {reference_sim} by {pos_diff}"
            )

            ref_joint_pos = reference_state.robots["robot"].joint_pos[0]
            comp_joint_pos = compare_state.robots["robot"].joint_pos[0]

            joint_diff = torch.norm(ref_joint_pos - comp_joint_pos).item()
            assert joint_diff < JOINT_TOLERANCE, (
                f"{sim_type} joint positions differ from {reference_sim} by {joint_diff}"
            )

    def test_motion_consistency(self, simulator_results):
        """Test that motion trajectories are consistent across simulators."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        sim_types = list(simulator_results.keys())
        reference_sim = sim_types[0]

        ref_final = simulator_results[reference_sim]["final_state"]
        ref_final_joints = ref_final.robots["robot"].joint_pos[0]

        for sim_type in sim_types[1:]:
            comp_final = simulator_results[sim_type]["final_state"]
            comp_final_joints = comp_final.robots["robot"].joint_pos[0]

            joint_diff = torch.norm(ref_final_joints - comp_final_joints).item()

            assert joint_diff < JOINT_TOLERANCE * 10, (
                f"{sim_type} final joint positions differ from {reference_sim} by {joint_diff}"
            )

    def test_object_physics_consistency(self, simulator_results):
        """Test that object physics behavior is consistent."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        sim_types = list(simulator_results.keys())
        reference_sim = sim_types[0]

        ref_final = simulator_results[reference_sim]["final_state"]
        ref_cube_pos = ref_final.objects["cube"].root_state[0, :3]

        for sim_type in sim_types[1:]:
            comp_final = simulator_results[sim_type]["final_state"]
            comp_cube_pos = comp_final.objects["cube"].root_state[0, :3]

            z_diff = abs(ref_cube_pos[2].item() - comp_cube_pos[2].item())
            assert z_diff < POSITION_TOLERANCE * 5, (
                f"{sim_type} cube Z position differs from {reference_sim} by {z_diff}"
            )

    def test_joint_name_consistency(self, simulator_results):
        """Test that joint names are consistent across simulators."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        joint_names_by_sim = {}

        for sim_name, results in simulator_results.items():
            handler = results["handler"]
            joint_names = handler.get_joint_names("robot", sort=True)
            joint_names_by_sim[sim_name] = joint_names

        reference_joints = list(joint_names_by_sim.values())[0]
        for sim_name, joints in joint_names_by_sim.items():
            assert joints == reference_joints, f"{sim_name} has different joint names: {joints} vs {reference_joints}"

    def test_state_format_consistency(self, simulator_results):
        """Test that state formats are consistent across simulators."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        for sim_name, results in simulator_results.items():
            state = results["initial_state"]

            assert isinstance(state, TensorState), f"{sim_name} doesn't return TensorState"

            robot_state = state.robots["robot"]
            assert hasattr(robot_state, "root_state")
            assert hasattr(robot_state, "joint_pos")
            assert hasattr(robot_state, "joint_vel")

            assert robot_state.root_state.shape[1] == 13, f"{sim_name} root_state has wrong shape"
            assert robot_state.joint_pos.shape[1] == 2, f"{sim_name} joint_pos has wrong shape"

    def test_action_format_acceptance(self, simulator_results, reference_scenario):
        """Test that all simulators accept the same action formats."""
        if len(simulator_results) < 2:
            pytest.skip("Not enough simulators available")

        dict_action = [{"robot": {"dof_pos_target": {"joint1": 0.3, "joint2": -0.3}}}]
        tensor_action = torch.tensor([[0.3, -0.3]], dtype=torch.float32)

        for sim_type, results in simulator_results.items():
            handler_class = get_sim_handler_class(sim_type)
            handler = handler_class(reference_scenario)
            handler.launch()

            try:
                handler.step(dict_action)
                dict_works = True
            except:
                dict_works = False

            try:
                if hasattr(handler, "device"):
                    tensor_action = tensor_action.to(handler.device)
                handler.step(tensor_action)
                tensor_works = True
            except:
                tensor_works = False

            handler.close()

            assert dict_works, f"{sim_type} doesn't accept dict action format"
            if not tensor_works:
                print(f"Warning: {sim_type} doesn't accept tensor action format")


@pytest.mark.parametrize("num_steps", [1, 10, 50])
def test_simulation_determinism(reference_scenario, num_steps):
    """Test that simulators produce deterministic results."""
    available_sims = get_available_simulators()

    for sim_type in available_sims[:2]:
        handler_class = get_sim_handler_class(sim_type)

        results = []
        for run in range(2):
            handler = handler_class(reference_scenario)
            handler.launch()

            action = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}}]

            for _ in range(num_steps):
                handler.step(action)

            final_state = handler._get_states()
            results.append(final_state)
            handler.close()

        state1 = results[0]
        state2 = results[1]

        joints1 = state1.robots["robot"].joint_pos
        joints2 = state2.robots["robot"].joint_pos

        assert torch.allclose(joints1, joints2, atol=1e-6), f"{sim_type} is not deterministic"


def test_error_handling_consistency(reference_scenario):
    """Test that error handling is consistent across simulators."""
    available_sims = get_available_simulators()

    for sim_type in available_sims[:2]:
        handler_class = get_sim_handler_class(sim_type)
        handler = handler_class(reference_scenario)
        handler.launch()

        with pytest.raises((KeyError, ValueError, RuntimeError)):
            invalid_action = [{"robot": {"dof_pos_target": {"invalid_joint": 0.5}}}]
            handler.step(invalid_action)

        handler.close()
