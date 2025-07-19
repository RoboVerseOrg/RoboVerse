"""MuJoCo simulator-specific tests.

Tests unique features and behaviors of the MuJoCo simulator while ensuring
it conforms to the base API.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg as CameraCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.sim.mujoco.mujoco import MujocoHandler, MujocoParallelHandler
from metasim.utils.state import TensorState

pytest.importorskip("mujoco")


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
        num_envs=1,
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


class TestMujocoHandler:
    """Test MuJoCo-specific functionality."""

    def test_single_env_constraint(self, simple_scenario):
        """Test that MuJoCo enforces single environment constraint."""
        handler = MujocoHandler(simple_scenario)
        assert handler.num_envs == 1

        simple_scenario.num_envs = 2
        with pytest.raises(ValueError, match="only supports single envs"):
            MujocoHandler(simple_scenario)

    def test_parallel_handler(self, simple_scenario):
        """Test MuJoCo parallel handler wrapper."""
        simple_scenario.num_envs = 4
        handler = MujocoParallelHandler(simple_scenario)
        handler.launch()

        assert handler.num_envs == 4

        actions = []
        for i in range(4):
            actions.append({"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}})

        obs, _, _, _, _ = handler.step(actions)
        assert isinstance(obs, TensorState)
        assert obs.robots["robot"].joint_pos.shape[0] == 4

        handler.close()

    def test_mjcf_path_usage(self, simple_scenario):
        """Test that MuJoCo uses mjcf_path when available."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        assert handler._robot_path == simple_scenario.robots[0].mjcf_path
        assert handler.robot_attached is not None

        handler.close()

    def test_manual_pd_control(self, simple_scenario):
        """Test manual PD control for effort-controlled joints."""
        simple_scenario.robots[0].control_type = {"joint1": "effort", "joint2": "effort"}

        handler = MujocoHandler(simple_scenario)
        handler.launch()

        assert handler._manual_pd_on == True
        assert handler._p_gains is not None
        assert handler._d_gains is not None
        assert len(handler._effort_controlled_joints) == 2
        assert len(handler._position_controlled_joints) == 0

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}}]
        handler.set_dof_targets("robot", actions)

        handler._simulate()

        handler.close()

    def test_gravity_compensation(self, simple_scenario):
        """Test gravity compensation feature."""
        simple_scenario.robots[0].enabled_gravity = False

        handler = MujocoHandler(simple_scenario)
        handler.launch()

        assert handler._gravity_compensation == True

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.0, "joint2": 0.0}}}]
        handler.set_dof_targets("robot", actions)
        handler._simulate()

        assert np.any(handler.physics.data.xfrc_applied != 0)

        handler.close()

    def test_decimation_handling(self, simple_scenario):
        """Test decimation parameter handling."""
        simple_scenario.decimation = 5

        handler = MujocoHandler(simple_scenario)
        handler.launch()

        assert handler.decimation == 5

        initial_time = handler.physics.data.time
        handler._simulate()
        final_time = handler.physics.data.time

        expected_time_advance = simple_scenario.sim_params.dt * handler.decimation
        actual_time_advance = final_time - initial_time

        assert abs(actual_time_advance - expected_time_advance) < 1e-6

        handler.close()

    def test_camera_rendering(self, simple_scenario):
        """Test camera rendering functionality."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        states = handler._get_states()

        assert "main_camera" in states.cameras
        camera_state = states.cameras["main_camera"]

        assert camera_state.rgb is not None
        assert camera_state.rgb.shape == (1, 480, 640, 3)
        assert camera_state.rgb.dtype == torch.uint8

        assert camera_state.depth is not None
        assert camera_state.depth.shape == (1, 480, 640)

        handler.close()

    def test_joint_reindexing(self, simple_scenario):
        """Test joint reindexing for sorted joint names."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        actuator_reindex = handler._get_actuator_reindex("robot")
        assert isinstance(actuator_reindex, list)

        body_reindex = handler._get_body_ids_reindex("robot")
        assert isinstance(body_reindex, list)

        handler.close()

    def test_action_caching(self, simple_scenario):
        """Test that actions are properly cached."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        actions = [{"robot": {"dof_pos_target": {"joint1": 0.5, "joint2": -0.5}}}]
        handler.set_dof_targets("robot", actions)

        assert handler.actions_cache == actions
        assert handler._actions_cache == actions

        handler.close()

    def test_primitive_object_creation(self, simple_scenario):
        """Test creation of primitive objects."""
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

        handler = MujocoHandler(simple_scenario)
        handler.launch()

        assert "cube" in handler.mj_objects
        assert "cylinder" in handler.mj_objects
        assert "sphere" in handler.mj_objects

        assert len(handler.object_body_names) == 3

        handler.close()

    def test_state_setting_with_velocity(self, simple_scenario):
        """Test setting states with velocity control."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        states = [
            {
                "robots": {
                    "robot": {
                        "pos": [0.0, 0.0, 0.0],
                        "rot": [1.0, 0.0, 0.0, 0.0],
                        "dof_pos": {"joint1": 0.5, "joint2": -0.5},
                    }
                },
                "objects": {
                    "cube": {
                        "pos": [0.5, 0.0, 0.5],
                        "rot": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }
        ]

        handler._set_states(states)

        cube_vel = handler.physics.data.joint("cube/").qvel
        assert np.allclose(cube_vel, 0.0)

        handler.close()

    def test_joint_limits_retrieval(self, simple_scenario):
        """Test getting joint limits."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler, "get_joint_limits"):
            limits = handler.get_joint_limits("robot", "joint1")
            assert len(limits) == 2

        handler.close()

    def test_api_compatibility_fixes(self, simple_scenario):
        """Test that MuJoCo handler has necessary fixes for API compatibility."""
        handler = MujocoHandler(simple_scenario)

        import inspect

        sig = inspect.signature(handler.step)
        param = sig.parameters["action"]

        handler.launch()
        states = handler._get_states()
        assert isinstance(states, TensorState), "MuJoCo should return TensorState from _get_states"

        handler.close()


@pytest.mark.parametrize("joint_mode", ["position", "effort", "mixed"])
def test_control_modes(simple_scenario, joint_mode):
    """Test different control modes."""
    if joint_mode == "position":
        control_type = {"joint1": "position", "joint2": "position"}
    elif joint_mode == "effort":
        control_type = {"joint1": "effort", "joint2": "effort"}
    else:
        control_type = {"joint1": "position", "joint2": "effort"}

    simple_scenario.robots[0].control_type = control_type

    handler = MujocoHandler(simple_scenario)
    handler.launch()

    if joint_mode == "effort" or joint_mode == "mixed":
        assert handler._manual_pd_on == True
    else:
        assert handler._manual_pd_on == False

    actions = [{"robot": {"dof_pos_target": {"joint1": 0.3, "joint2": -0.3}}}]
    handler.set_dof_targets("robot", actions)
    handler._simulate()

    handler.close()


def test_scale_application(simple_scenario):
    """Test scale application to MJCF models."""
    scaled_cube = ArticulationObjCfg(
        name="scaled_cube",
        mjcf_path=str(Path(__file__).parent.parent / "assets" / "objects" / "articulated_box.xml"),
        urdf_path=str(Path(__file__).parent.parent / "assets" / "objects" / "articulated_box.urdf"),
        scale=(2.0, 1.5, 1.0),
        fix_base_link=False,
    )
    simple_scenario.objects.append(scaled_cube)

    handler = MujocoHandler(simple_scenario)
    handler.launch()

    assert "scaled_cube" in handler.mj_objects

    handler.close()


@pytest.mark.mujoco
class TestMuJoCoAdvancedFeatures:
    """Test advanced MuJoCo-specific features."""

    def test_mujoco_contact_forces(self, simple_scenario):
        """Test MuJoCo's contact force reporting."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        handler.reset()
        for _ in range(50):
            handler._simulate()

        if hasattr(handler.physics.data, "contact"):
            contacts = handler.physics.data.contact
            assert len(contacts) > 0 or handler.physics.data.ncon > 0

        handler.close()

    def test_mujoco_solver_parameters(self, simple_scenario):
        """Test MuJoCo solver parameter access."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics.model, "opt"):
            solver_opt = handler.physics.model.opt
            original_iterations = solver_opt.iterations

            solver_opt.iterations = 10
            assert solver_opt.iterations == 10

            solver_opt.iterations = original_iterations

        handler.close()

    def test_mujoco_sensor_data(self, simple_scenario):
        """Test MuJoCo sensor data access."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics.data, "sensordata"):
            sensor_data = handler.physics.data.sensordata
            assert isinstance(sensor_data, np.ndarray)

        handler.close()

    def test_mujoco_jacobian_computation(self, simple_scenario):
        """Test MuJoCo's Jacobian computation capabilities."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics.data, "get_body_jacp"):
            body_id = handler.physics.model.body(handler._mujoco_robot_name + "link1").id
            jacp = np.zeros((3, handler.physics.model.nv))
            handler.physics.data.get_body_jacp(jacp, body_id)

            assert isinstance(jacp, np.ndarray)
            assert jacp.shape[0] == 3

        handler.close()

    def test_mujoco_forward_dynamics(self, simple_scenario):
        """Test MuJoCo's forward dynamics computation."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        joint_names = handler.get_joint_names("robot", sort=True)
        controls = np.array([1.0, 0.5])

        for i, joint_name in enumerate(joint_names):
            actuator_id = handler.physics.model.actuator(f"{handler._mujoco_robot_name}{joint_name}").id
            handler.physics.data.ctrl[actuator_id] = controls[i]

        handler.physics.forward()

        qacc = handler.physics.data.qacc
        assert isinstance(qacc, np.ndarray)
        assert len(qacc) >= len(controls)

        handler.close()

    def test_mujoco_inverse_dynamics(self, simple_scenario):
        """Test MuJoCo's inverse dynamics computation."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics, "inverse"):
            handler.physics.data.qacc[:] = 0.1

            handler.physics.inverse()

            qfrc_inverse = handler.physics.data.qfrc_inverse
            assert isinstance(qfrc_inverse, np.ndarray)
            assert np.any(np.abs(qfrc_inverse) > 1e-6)

        handler.close()

    def test_mujoco_warmstart(self, simple_scenario):
        """Test MuJoCo's warmstart capabilities."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics.model.opt, "enableflags"):
            import mujoco

            if hasattr(mujoco, "mjtEnableBit") and hasattr(mujoco.mjtEnableBit, "mjENBL_WARMSTART"):
                handler.physics.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_WARMSTART

                for _ in range(10):
                    handler._simulate()

        handler.close()

    def test_mujoco_mass_matrix(self, simple_scenario):
        """Test MuJoCo's mass matrix computation."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if hasattr(handler.physics.data, "get_body_mass"):
            body_id = handler.physics.model.body(handler._mujoco_robot_name).id
            mass = handler.physics.data.get_body_mass()[body_id]
            assert mass > 0

        if hasattr(handler.physics, "fullM"):
            import mujoco

            M = np.zeros((handler.physics.model.nv, handler.physics.model.nv))
            mujoco.mj_fullM(handler.physics.model.ptr, M, handler.physics.data.qM)
            assert isinstance(M, np.ndarray)
            assert M.shape[0] == M.shape[1]
            assert np.allclose(M, M.T)

        handler.close()

    def test_mujoco_visual_geom_groups(self, simple_scenario):
        """Test MuJoCo's visual geometry group management."""
        handler = MujocoHandler(simple_scenario)
        handler.launch()

        if not handler.headless and handler.viewer is not None:
            if hasattr(handler.viewer, "vopt"):
                for i in range(5):
                    if hasattr(handler.viewer.vopt, "geomgroup"):
                        handler.viewer.vopt.geomgroup[i] = 0
                        handler.viewer.vopt.geomgroup[i] = 1

        handler.close()
