"""Test replay_demo.py functionality across all simulators."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.utils.state import TensorState

ALL_SIMULATORS = ["isaacgym", "isaaclab", "genesis", "mujoco", "sapien2", "sapien3", "pybullet", "mjx"]

TEST_TASKS = ["StackCube", "PickCube", "ReachTarget"]

TEST_CONFIGS = [
    {"num_envs": 1},
    {"num_envs": 4},
    {"num_envs": 8},
]


@pytest.fixture
def mock_trajectory_data():
    """Create mock trajectory data for testing."""
    num_envs = 4
    num_steps = 10

    init_states = TensorState(
        joint_pos=torch.randn(num_envs, 7),
        joint_vel=torch.zeros(num_envs, 7),
        root_pos=torch.zeros(num_envs, 3),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs),
        root_vel=torch.zeros(num_envs, 6),
    )

    all_actions = []
    for env_idx in range(num_envs):
        env_actions = []
        for step in range(num_steps):
            action = torch.randn(7) * 0.1
            env_actions.append(action)
        all_actions.append(env_actions)

    all_states = []
    for env_idx in range(num_envs):
        env_states = []
        for step in range(num_steps):
            state = TensorState(
                joint_pos=torch.randn(7),
                joint_vel=torch.zeros(7),
                root_pos=torch.zeros(3),
                root_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                root_vel=torch.zeros(6),
            )
            env_states.append(state)
        all_states.append(env_states)

    return init_states, all_actions, all_states


@pytest.fixture
def create_mock_trajectory_file(tmp_path):
    """Create a mock trajectory file for testing."""
    import pickle

    def _create_file(task_name, robot_name="franka"):
        traj_dir = tmp_path / "trajs" / task_name / "v2"
        traj_dir.mkdir(parents=True, exist_ok=True)

        traj_file = traj_dir / f"{robot_name}_v2.pkl"

        num_steps = 10
        mock_data = {
            robot_name: {
                "init_state": {
                    "joint_pos": np.random.randn(7).tolist(),
                    "joint_vel": np.zeros(7).tolist(),
                },
                "actions": [np.random.randn(7).tolist() for _ in range(num_steps)],
                "states": [
                    {
                        "joint_pos": np.random.randn(7).tolist(),
                        "joint_vel": np.zeros(7).tolist(),
                    }
                    for _ in range(num_steps)
                ],
            }
        }

        with open(traj_file, "wb") as f:
            pickle.dump(mock_data, f)

        return str(traj_dir)

    return _create_file


class TestReplayDemoUnit:
    """Unit tests for replay_demo.py components."""

    def test_get_actions(self):
        """Test action retrieval logic."""
        from metasim.cfg.robots.franka_cfg import FrankaCfg
        from metasim.scripts.replay_demo import get_actions

        robot = FrankaCfg()
        all_actions = [
            [torch.tensor([1.0] * 7), torch.tensor([2.0] * 7)],
            [torch.tensor([3.0] * 7), torch.tensor([4.0] * 7)],
        ]

        actions = get_actions(all_actions, 0, 2, robot)
        assert len(actions) == 2
        assert torch.allclose(actions[0], torch.tensor([1.0] * 7))
        assert torch.allclose(actions[1], torch.tensor([3.0] * 7))

        actions = get_actions(all_actions, 5, 2, robot)
        assert torch.allclose(actions[0], torch.tensor([2.0] * 7))
        assert torch.allclose(actions[1], torch.tensor([4.0] * 7))

    def test_get_states(self):
        """Test state retrieval logic."""
        from metasim.scripts.replay_demo import get_states

        state1 = TensorState(joint_pos=torch.tensor([1.0] * 7))
        state2 = TensorState(joint_pos=torch.tensor([2.0] * 7))
        state3 = TensorState(joint_pos=torch.tensor([3.0] * 7))
        state4 = TensorState(joint_pos=torch.tensor([4.0] * 7))

        all_states = [
            [state1, state2],
            [state3, state4],
        ]

        states = get_states(all_states, 0, 2)
        assert len(states) == 2
        assert torch.allclose(states[0].joint_pos, torch.tensor([1.0] * 7))
        assert torch.allclose(states[1].joint_pos, torch.tensor([3.0] * 7))

        states = get_states(all_states, 5, 2)
        assert torch.allclose(states[0].joint_pos, torch.tensor([2.0] * 7))
        assert torch.allclose(states[1].joint_pos, torch.tensor([4.0] * 7))

    def test_get_runout(self):
        """Test runout detection."""
        from metasim.scripts.replay_demo import get_runout

        all_actions = [
            [1, 2, 3],
            [1, 2],
            [1, 2, 3, 4],
        ]

        assert not get_runout(all_actions, 1)
        assert not get_runout(all_actions, 2)
        assert get_runout(all_actions, 4)

    def test_obs_saver(self, tmp_path):
        """Test ObsSaver functionality."""
        from metasim.scripts.replay_demo import ObsSaver

        image_dir = str(tmp_path / "images")
        saver = ObsSaver(image_dir=image_dir)

        camera_data = MagicMock()
        camera_data.rgb = torch.rand(4, 64, 64, 3) * 255

        state = TensorState()
        state.cameras = {"camera": camera_data}

        saver.add(state)
        saver.add(state)

        assert len(list(Path(image_dir).glob("*.png"))) == 2

        video_path = str(tmp_path / "video.mp4")
        saver = ObsSaver(video_path=video_path)
        saver.add(state)
        saver.add(state)
        saver.save()


class TestReplayDemoIntegration:
    """Integration tests for replay_demo.py across simulators."""

    @pytest.mark.parametrize("simulator", ALL_SIMULATORS)
    @pytest.mark.parametrize("num_envs", [1, 4])
    def test_replay_demo_mock_data(self, simulator, num_envs, mock_trajectory_data, tmp_path):
        """Test replay demo with mock trajectory data."""
        if simulator in ["isaacgym", "isaaclab", "genesis"]:
            pytest.skip(f"{simulator} requires GPU/special setup")

        init_states, all_actions, all_states = mock_trajectory_data

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env:
            with patch("metasim.scripts.replay_demo.get_traj") as mock_get_traj:
                mock_env = MagicMock()
                mock_handler = MagicMock()
                mock_env.handler = mock_handler
                mock_env.reset.return_value = (init_states, {})
                mock_env.step.return_value = (
                    init_states,
                    torch.zeros(num_envs),
                    torch.zeros(num_envs, dtype=torch.bool),
                    torch.zeros(num_envs, dtype=torch.bool),
                    {},
                )

                mock_env_class = MagicMock(return_value=mock_env)
                mock_get_env.return_value = mock_env_class
                mock_get_traj.return_value = (init_states, all_actions, all_states)

                camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
                scenario = ScenarioCfg(
                    task="StackCube",
                    robots=["franka"],
                    cameras=[camera],
                    sim=simulator,
                    num_envs=num_envs,
                    headless=True,
                )

                from metasim.scripts import replay_demo

                with patch.object(replay_demo, "args") as mock_args:
                    mock_args.task = "StackCube"
                    mock_args.robot = "franka"
                    mock_args.sim = simulator
                    mock_args.num_envs = num_envs
                    mock_args.headless = True
                    mock_args.save_image_dir = None
                    mock_args.save_video_path = None
                    mock_args.stop_on_runout = True
                    mock_args.object_states = False
                    mock_args.renderer = None
                    mock_args.scene = None
                    mock_args.try_add_table = True
                    mock_args.split = "all"

                    mock_task = MagicMock()
                    mock_task.traj_filepath = str(tmp_path / "fake_traj.pkl")

                    with patch.object(scenario, "task", mock_task):
                        with patch("os.path.exists", return_value=True):
                            replay_demo.main()

                mock_get_env.assert_called_once()
                mock_env.reset.assert_called_once()
                assert mock_env.step.call_count > 0
                mock_env.close.assert_called_once()

    @pytest.mark.parametrize("simulator", ["mujoco", "sapien2", "pybullet"])
    def test_replay_demo_rendering_modes(self, simulator, create_mock_trajectory_file, tmp_path):
        """Test different rendering configurations."""
        traj_path = create_mock_trajectory_file("test_task", "franka")

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env:
            mock_env = MagicMock()
            mock_handler = MagicMock()
            mock_env.handler = mock_handler

            camera_rgb = torch.rand(4, 64, 64, 3) * 255
            camera_mock = MagicMock()
            camera_mock.rgb = camera_rgb
            obs_state = TensorState()
            obs_state.cameras = {"camera": camera_mock}

            mock_env.reset.return_value = (obs_state, {})
            mock_env.step.return_value = (
                obs_state,
                torch.zeros(4),
                torch.ones(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool),
                {},
            )

            mock_env_class = MagicMock(return_value=mock_env)
            mock_get_env.return_value = mock_env_class

            image_dir = str(tmp_path / "test_images")

            from metasim.scripts import replay_demo

            with patch.object(replay_demo, "args") as mock_args:
                mock_args.task = "test_task"
                mock_args.robot = "franka"
                mock_args.sim = simulator
                mock_args.num_envs = 4
                mock_args.headless = True
                mock_args.save_image_dir = image_dir
                mock_args.save_video_path = None
                mock_args.stop_on_runout = False
                mock_args.object_states = False
                mock_args.renderer = None
                mock_args.scene = None
                mock_args.try_add_table = True
                mock_args.split = "all"

                mock_scenario = MagicMock()
                mock_scenario.task.traj_filepath = traj_path
                mock_scenario.sim = simulator
                mock_scenario.renderer = None
                mock_scenario.num_envs = 4
                mock_scenario.robots = [MagicMock()]

                with patch("metasim.scripts.replay_demo.ScenarioCfg", return_value=mock_scenario):
                    replay_demo.main()

                assert os.path.exists(image_dir)

    @pytest.mark.parametrize("simulator", ["mujoco", "genesis"])
    def test_replay_demo_hybrid_rendering(self, simulator, create_mock_trajectory_file):
        """Test hybrid rendering (different physics and rendering simulators)."""
        traj_path = create_mock_trajectory_file("test_task", "franka")

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env:
            with patch("metasim.scripts.replay_demo.HybridSimEnv") as mock_hybrid:
                mock_physics_env = MagicMock()
                mock_render_env = MagicMock()
                mock_hybrid_env = MagicMock()

                mock_handler = MagicMock()
                mock_hybrid_env.handler = mock_handler

                obs_state = TensorState()
                mock_hybrid_env.reset.return_value = (obs_state, {})
                mock_hybrid_env.step.return_value = (
                    obs_state,
                    torch.zeros(4),
                    torch.ones(4, dtype=torch.bool),
                    torch.zeros(4, dtype=torch.bool),
                    {},
                )

                mock_get_env.return_value = MagicMock(side_effect=[mock_render_env, mock_physics_env])
                mock_hybrid.return_value = mock_hybrid_env

                from metasim.scripts import replay_demo

                with patch.object(replay_demo, "args") as mock_args:
                    mock_args.task = "test_task"
                    mock_args.robot = "franka"
                    mock_args.sim = simulator
                    mock_args.renderer = "isaaclab" if simulator != "isaaclab" else "mujoco"
                    mock_args.num_envs = 4
                    mock_args.headless = True
                    mock_args.save_image_dir = None
                    mock_args.save_video_path = None
                    mock_args.stop_on_runout = False
                    mock_args.object_states = False
                    mock_args.scene = None
                    mock_args.try_add_table = True
                    mock_args.split = "all"

                    mock_scenario = MagicMock()
                    mock_scenario.task.traj_filepath = traj_path
                    mock_scenario.sim = simulator
                    mock_scenario.renderer = mock_args.renderer
                    mock_scenario.num_envs = 4
                    mock_scenario.robots = [MagicMock()]

                    with patch("metasim.scripts.replay_demo.ScenarioCfg", return_value=mock_scenario):
                        replay_demo.main()

                mock_hybrid.assert_called_once()
                mock_hybrid_env.reset.assert_called_once()
                mock_hybrid_env.close.assert_called_once()


class TestReplayDemoCommand:
    """Test replay_demo.py as a command-line script."""

    @pytest.mark.parametrize("simulator", ["mujoco", "sapien2"])
    def test_command_line_execution(self, simulator, create_mock_trajectory_file, tmp_path):
        """Test running replay_demo.py via command line."""
        traj_path = create_mock_trajectory_file("test_task", "franka")

        test_script = tmp_path / "test_replay.py"
        test_script.write_text(f"""
import sys
import os
from unittest.mock import patch, MagicMock
import torch

sys.path.insert(0, os.path.abspath('.'))

with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env:
    mock_env = MagicMock()
    mock_handler = MagicMock()
    mock_env.handler = mock_handler

    obs_state = MagicMock()
    obs_state.cameras = {{}}

    mock_env.reset.return_value = (obs_state, {{}})
    mock_env.step.return_value = (
        obs_state,
        torch.zeros(4),
        torch.ones(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
        {{}}
    )

    mock_env_class = MagicMock(return_value=mock_env)
    mock_get_env.return_value = mock_env_class

    with patch("metasim.cfg.tasks.get_task_cfg") as mock_get_task:
        mock_task = MagicMock()
        mock_task.traj_filepath = "{traj_path}"
        mock_get_task.return_value = mock_task

        import metasim.scripts.replay_demo

print("SUCCESS")
""")

        cmd = [
            sys.executable,
            str(test_script),
            f"--sim={simulator}",
            "--task=test_task",
            "--num_envs=4",
            "--robot=franka",
            "--headless",
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(Path.cwd()))

        assert "SUCCESS" in result.stdout or result.returncode == 0


@pytest.mark.slow
class TestReplayDemoRealData:
    """Test with real trajectory data if available."""

    @pytest.mark.parametrize("simulator", ["mujoco"])
    def test_with_real_trajectory(self, simulator):
        """Test with actual trajectory files if they exist."""
        traj_path = Path("metasim/data/quick_start/trajs/rlbench/close_box/v2")
        if not traj_path.exists():
            pytest.skip("Real trajectory data not available")

        pytest.skip("Real simulator integration test - requires full setup")


def test_all_simulators_supported():
    """Ensure all simulators are properly defined in the test."""
    from metasim.constants import SimType

    all_sim_types = [sim.value for sim in SimType]

    special_types = ["hybrid", "blender"]
    actual_sims = [s for s in all_sim_types if s not in special_types]

    missing = set(actual_sims) - set(ALL_SIMULATORS)
    extra = set(ALL_SIMULATORS) - set(actual_sims)

    assert not missing, f"Missing simulators in test: {missing}"
    assert not extra, f"Extra simulators in test: {extra}"
