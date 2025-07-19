"""Test configuration and fixtures for metasim unit tests.

This module provides pytest fixtures and configuration for testing simulator handlers.
It includes parametrized fixtures that allow tests to run across all available simulators.
"""

import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ["ROBOVERSE_NO_DOWNLOAD"] = "1"

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TEST_ASSETS_DIR = Path(__file__).parent / "assets"


class SimulatorNotAvailableError(Exception):
    """Raised when a simulator is not available for testing."""

    pass


def check_pybullet_available():
    """Check if PyBullet is available."""
    try:
        import pybullet

        return True
    except ImportError:
        return False


def check_mujoco_available():
    """Check if MuJoCo is available."""
    try:
        import mujoco

        return True
    except ImportError:
        return False


def check_isaacgym_available():
    """Check if Isaac Gym is available."""
    try:
        import isaacgym
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def check_isaaclab_available():
    """Check if Isaac Lab is available."""
    try:
        import omni.isaac.lab

        return True
    except ImportError:
        return False


def check_sapien_available():
    """Check if SAPIEN is available."""
    try:
        import sapien

        return True
    except ImportError:
        return False


def check_genesis_available():
    """Check if Genesis is available."""
    try:
        import genesis

        return True
    except ImportError:
        return False


def check_mjx_available():
    """Check if MJX is available."""
    try:
        import jax
        import mujoco

        try:
            gpu_devices = jax.devices("gpu")
            return len(gpu_devices) > 0
        except RuntimeError:
            return False
    except ImportError:
        return False


def check_pyrep_available():
    """Check if PyRep is available."""
    try:
        import pyrep

        return True
    except ImportError:
        return False


SIMULATOR_CHECKERS = {
    "pybullet": check_pybullet_available,
    "mujoco": check_mujoco_available,
    "isaacgym": check_isaacgym_available,
    "isaaclab": check_isaaclab_available,
    "sapien": check_sapien_available,
    "genesis": check_genesis_available,
    "mjx": check_mjx_available,
    "pyrep": check_pyrep_available,
}


def create_handler(simulator_name: str, scenario_cfg=None, **kwargs):
    """Create a simulator handler instance.

    Args:
        simulator_name: Name of the simulator
        scenario_cfg: ScenarioCfg object (if None, creates a basic one)
        **kwargs: Additional arguments for scenario creation if scenario_cfg is None

    Returns:
        Initialized simulator handler

    Raises:
        SimulatorNotAvailableError: If simulator is not available
    """
    if not SIMULATOR_CHECKERS[simulator_name]():
        raise SimulatorNotAvailableError(f"Simulator '{simulator_name}' is not available")

    if scenario_cfg is None:
        from pathlib import Path

        from metasim.cfg.checkers import EmptyChecker
        from metasim.cfg.robots import BaseRobotCfg
        from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg
        from metasim.cfg.scenario import ScenarioCfg
        from metasim.cfg.simulator_params import SimParamCfg

        num_envs = kwargs.get("num_envs", 1)
        headless = kwargs.get("headless", True)

        robot = BaseRobotCfg(
            name="test_robot",
            urdf_path=str(Path(__file__).parent / "assets" / "robots" / "simple_arm.urdf"),
            mjcf_path=str(Path(__file__).parent / "assets" / "robots" / "simple_arm.xml"),
            fix_base_link=True,
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

        scenario_cfg = ScenarioCfg(
            num_envs=num_envs,
            robots=[robot],
            objects=[],
            cameras=[],
            checker=EmptyChecker(),
            sim_params=SimParamCfg(dt=0.01),
            decimation=1,
            episode_length=100,
            headless=headless,
        )

    try:
        if simulator_name == "pybullet":
            from metasim.sim.pybullet.pybullet import SinglePybulletHandler

            return SinglePybulletHandler(scenario_cfg)
        elif simulator_name == "mujoco":
            from metasim.sim.mujoco.mujoco import MujocoHandler

            return MujocoHandler(scenario_cfg)
        elif simulator_name == "isaacgym":
            from metasim.sim.isaacgym.isaacgym import IsaacgymHandler

            return IsaacgymHandler(scenario_cfg)
        elif simulator_name == "isaaclab":
            from metasim.sim.isaaclab.isaaclab import IsaaclabHandler

            return IsaaclabHandler(scenario_cfg)
        elif simulator_name == "sapien":
            from metasim.sim.sapien.sapien2 import Sapien2Handler

            return Sapien2Handler(scenario_cfg)
        elif simulator_name == "genesis":
            from metasim.sim.genesis.genesis import GenesisHandler

            return GenesisHandler(scenario_cfg)
        elif simulator_name == "mjx":
            from metasim.sim.mjx.mjx import MJXHandler

            return MJXHandler(scenario_cfg)
        elif simulator_name == "pyrep":
            from metasim.sim.pyrep.pyrep import PyrepHandler

            return PyrepHandler(scenario_cfg)
        else:
            raise ValueError(f"Unknown simulator: {simulator_name}")
    except ImportError as e:
        raise SimulatorNotAvailableError(f"Failed to import {simulator_name} handler: {e}")


AVAILABLE_SIMULATORS = []
for sim_name, checker in SIMULATOR_CHECKERS.items():
    if checker():
        AVAILABLE_SIMULATORS.append(sim_name)
        logger.info(f"Simulator '{sim_name}' is available for testing")
    else:
        logger.info(f"Simulator '{sim_name}' is NOT available for testing")


@pytest.fixture(params=AVAILABLE_SIMULATORS)
def simulator_handler(request, tmp_path):
    """Parametrized fixture that yields an initialized handler for each available backend.

    This fixture will cause any test using it to run once for each available simulator.
    Tests can access the simulator name via request.param.
    """
    simulator_name = request.param
    logger.info(f"Creating handler for simulator: {simulator_name}")

    try:
        handler = create_handler(
            simulator_name,
            num_envs=1,
            device="cpu",
            headless=True,
            assets_dir=str(TEST_ASSETS_DIR),
            tmp_dir=str(tmp_path),
        )

        handler.launch()

        yield handler

    except Exception as e:
        logger.error(f"Failed to create handler for {simulator_name}: {e}")
        pytest.skip(f"Could not initialize {simulator_name}: {e}")
    finally:
        try:
            if "handler" in locals():
                handler.close()
        except Exception as e:
            logger.error(f"Error closing handler: {e}")


@pytest.fixture
def mock_handler():
    """Create a mock simulator handler for pure unit tests."""
    handler = Mock()

    handler.num_envs = 1
    handler.device = "cpu"
    handler.simulator_name = "mock"

    handler.get_joint_names.return_value = ["joint1", "joint2", "joint3"]
    handler.get_body_names.return_value = ["base", "link1", "link2", "link3"]
    handler.get_states.return_value = {
        "joint_positions": np.zeros(3),
        "joint_velocities": np.zeros(3),
        "base_position": np.zeros(3),
        "base_orientation": np.array([1.0, 0.0, 0.0, 0.0]),
    }

    return handler


@pytest.fixture
def simple_robot_urdf(tmp_path):
    """Create a simple robot URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>
</robot>"""

    urdf_path = tmp_path / "simple_arm.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def simple_cube_urdf(tmp_path):
    """Create a simple cube URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
</robot>"""

    urdf_path = tmp_path / "cube.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def performance_monitor():
    """Fixture for monitoring performance metrics during tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_start = None
            self.memory_end = None
            self.metrics = {}

        def start(self):
            """Start monitoring."""
            import os

            import psutil

            self.start_time = time.time()
            process = psutil.Process(os.getpid())
            self.memory_start = process.memory_info().rss / 1024 / 1024

        def stop(self):
            """Stop monitoring and calculate metrics."""
            import os

            import psutil

            self.end_time = time.time()
            process = psutil.Process(os.getpid())
            self.memory_end = process.memory_info().rss / 1024 / 1024

            self.metrics = {
                "duration": self.end_time - self.start_time,
                "memory_used": self.memory_end - self.memory_start,
                "memory_peak": self.memory_end,
            }

            return self.metrics

        def assert_performance(self, max_duration=None, max_memory=None):
            """Assert performance metrics are within bounds."""
            if max_duration and self.metrics["duration"] > max_duration:
                pytest.fail(f"Test took {self.metrics['duration']}s, expected < {max_duration}s")

            if max_memory and self.metrics["memory_used"] > max_memory:
                pytest.fail(f"Test used {self.metrics['memory_used']}MB, expected < {max_memory}MB")

    return PerformanceMonitor()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "unit: marks pure unit tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "performance: marks performance tests")

    for sim in SIMULATOR_CHECKERS.keys():
        config.addinivalue_line("markers", f"{sim}: marks tests specific to {sim}")


POSITION_TOLERANCE = 1e-3
ORIENTATION_TOLERANCE = 1e-3
VELOCITY_TOLERANCE = 1e-2
FORCE_TOLERANCE = 1e-1


@pytest.fixture
def tolerances():
    """Fixture providing standard tolerance values for comparisons."""
    return {
        "position": POSITION_TOLERANCE,
        "orientation": ORIENTATION_TOLERANCE,
        "velocity": VELOCITY_TOLERANCE,
        "force": FORCE_TOLERANCE,
    }


@pytest.fixture
def create_mock_trajectory_file(tmp_path):
    """Create a mock trajectory file for replay demo testing."""
    import pickle

    def _create_file(task_name, robot_name="franka", num_envs=4, num_steps=10):
        traj_dir = tmp_path / "trajs" / task_name / "v2"
        traj_dir.mkdir(parents=True, exist_ok=True)

        traj_file = traj_dir / f"{robot_name}_v2.pkl"

        mock_data = {
            robot_name: {
                "init_state": {
                    "joint_pos": np.random.randn(num_envs, 7).tolist(),
                    "joint_vel": np.zeros((num_envs, 7)).tolist(),
                    "root_pos": np.zeros((num_envs, 3)).tolist(),
                    "root_quat": np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1)).tolist(),
                    "root_vel": np.zeros((num_envs, 6)).tolist(),
                },
                "actions": [[np.random.randn(7).tolist() for _ in range(num_steps)] for _ in range(num_envs)],
                "states": [
                    [
                        {
                            "joint_pos": np.random.randn(7).tolist(),
                            "joint_vel": np.zeros(7).tolist(),
                            "root_pos": np.zeros(3).tolist(),
                            "root_quat": [1.0, 0.0, 0.0, 0.0],
                            "root_vel": np.zeros(6).tolist(),
                        }
                        for _ in range(num_steps)
                    ]
                    for _ in range(num_envs)
                ],
            }
        }

        with open(traj_file, "wb") as f:
            pickle.dump(mock_data, f)

        return str(traj_dir)

    return _create_file


@pytest.fixture
def mock_camera_observation():
    """Create mock camera observation data."""
    import torch

    from metasim.utils.state import TensorState

    def _create_obs(num_envs=4, width=128, height=128, num_cameras=1):
        state = TensorState()
        state.cameras = {}

        for i in range(num_cameras):
            camera_name = f"camera_{i}" if i > 0 else "camera"
            camera_data = type(
                "CameraData",
                (),
                {
                    "rgb": torch.rand(num_envs, height, width, 3) * 255,
                    "depth": torch.rand(num_envs, height, width, 1),
                    "segmentation": torch.randint(0, 10, (num_envs, height, width, 1)),
                },
            )()
            state.cameras[camera_name] = camera_data

        return state

    return _create_obs


@pytest.fixture
def replay_demo_args():
    """Create mock args for replay_demo.py."""
    from metasim.cfg.randomization import RandomizationCfg
    from metasim.cfg.render import RenderCfg

    class MockArgs:
        task = "StackCube"
        robot = "franka"
        scene = None
        render = RenderCfg()
        random = RandomizationCfg()
        sim = "mujoco"
        renderer = None
        num_envs = 4
        try_add_table = True
        object_states = False
        split = "all"
        headless = True
        save_image_dir = None
        save_video_path = None
        stop_on_runout = False

    return MockArgs()


@pytest.fixture
def mock_sim_env(mock_camera_observation):
    """Create a mock simulation environment for testing."""
    from unittest.mock import MagicMock

    import torch

    def _create_env(num_envs=4, auto_success_after=5):
        env = MagicMock()
        handler = MagicMock()
        env.handler = handler

        env._step_count = 0

        initial_obs = mock_camera_observation(num_envs=num_envs)
        env.reset.return_value = (initial_obs, {})

        def mock_step(actions):
            env._step_count += 1
            obs = mock_camera_observation(num_envs=num_envs)
            reward = torch.zeros(num_envs)
            success = (
                torch.ones(num_envs, dtype=torch.bool)
                if env._step_count >= auto_success_after
                else torch.zeros(num_envs, dtype=torch.bool)
            )
            timeout = torch.zeros(num_envs, dtype=torch.bool)
            extras = {}
            return obs, reward, success, timeout, extras

        env.step = mock_step
        handler.num_envs = num_envs

        return env

    return _create_env
