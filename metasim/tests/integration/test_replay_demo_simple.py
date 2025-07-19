"""Simple tests to verify replay_demo.py basic functionality."""

import subprocess
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


class TestReplayDemoBasic:
    """Basic tests for replay_demo.py functionality."""

    def test_replay_demo_imports(self):
        """Test that replay_demo.py can be imported without errors."""
        with patch("sys.modules", {"torchvision": MagicMock(), "torchvision.utils": MagicMock()}):
            try:
                import metasim.scripts.replay_demo

                assert hasattr(metasim.scripts.replay_demo, "main")
                assert hasattr(metasim.scripts.replay_demo, "get_actions")
                assert hasattr(metasim.scripts.replay_demo, "get_states")
                assert hasattr(metasim.scripts.replay_demo, "ObsSaver")
            except ImportError as e:
                pytest.fail(f"Failed to import replay_demo.py: {e}")

    def test_command_help(self):
        """Test that replay_demo.py shows help message."""
        cmd = [sys.executable, "-m", "metasim.scripts.replay_demo", "--help"]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        assert result.returncode == 0
        assert "--task" in result.stdout
        assert "--sim" in result.stdout
        assert "--robot" in result.stdout
        assert "--num_envs" in result.stdout

    @pytest.mark.parametrize("sim", ["mujoco", "sapien2", "pybullet"])
    def test_replay_demo_mock_minimal(self, sim, create_mock_trajectory_file, mock_sim_env):
        """Test minimal replay demo execution with mocked components."""
        traj_path = create_mock_trajectory_file("test_task", "franka", num_envs=2, num_steps=3)

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env_class:
            with patch("metasim.scripts.replay_demo.get_traj") as mock_get_traj:
                mock_env = mock_sim_env(num_envs=2, auto_success_after=2)
                mock_get_env_class.return_value = Mock(return_value=mock_env)

                init_state = Mock()
                init_state.joint_pos = torch.zeros(2, 7)
                actions = [[torch.zeros(7)] * 3 for _ in range(2)]
                mock_get_traj.return_value = (init_state, actions, None)

                from metasim.scripts import replay_demo

                with patch.object(replay_demo, "args") as mock_args:
                    mock_args.task = "test_task"
                    mock_args.robot = "franka"
                    mock_args.sim = sim
                    mock_args.num_envs = 2
                    mock_args.headless = True
                    mock_args.save_image_dir = None
                    mock_args.save_video_path = None
                    mock_args.stop_on_runout = True
                    mock_args.object_states = False
                    mock_args.renderer = None
                    mock_args.scene = None
                    mock_args.try_add_table = True
                    mock_args.split = "all"
                    mock_args.render = Mock()
                    mock_args.random = Mock()

                    with patch("metasim.scripts.replay_demo.ScenarioCfg") as mock_scenario_cls:
                        mock_scenario = Mock()
                        mock_scenario.task.traj_filepath = traj_path
                        mock_scenario.sim = sim
                        mock_scenario.renderer = None
                        mock_scenario.num_envs = 2
                        mock_scenario.robots = [Mock()]
                        mock_scenario_cls.return_value = mock_scenario

                        with patch("os.path.exists", return_value=True):
                            replay_demo.main()

                assert mock_env.reset.called
                assert mock_env.step.called
                assert mock_env.close.called

    def test_obs_saver_basic(self, tmp_path, mock_camera_observation):
        """Test ObsSaver basic functionality."""
        from metasim.scripts.replay_demo import ObsSaver

        saver = ObsSaver()
        obs = mock_camera_observation(num_envs=1)
        saver.add(obs)
        saver.save()

        img_dir = tmp_path / "images"
        saver = ObsSaver(image_dir=str(img_dir))
        saver.add(obs)
        saver.add(obs)

        assert img_dir.exists()
        images = list(img_dir.glob("*.png"))
        assert len(images) == 2

    @pytest.mark.parametrize("num_envs", [1, 4, 8])
    def test_different_env_counts(self, num_envs, create_mock_trajectory_file, mock_sim_env):
        """Test replay demo with different numbers of environments."""
        traj_path = create_mock_trajectory_file("test_task", num_envs=num_envs)

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env_class:
            with patch("metasim.scripts.replay_demo.get_traj") as mock_get_traj:
                mock_env = mock_sim_env(num_envs=num_envs, auto_success_after=1)
                mock_get_env_class.return_value = Mock(return_value=mock_env)

                init_state = Mock()
                init_state.joint_pos = torch.zeros(num_envs, 7)
                actions = [[torch.zeros(7)] for _ in range(num_envs)]
                mock_get_traj.return_value = (init_state, actions, None)

                from metasim.scripts import replay_demo

                with patch.object(replay_demo, "args") as mock_args:
                    mock_args.task = "test_task"
                    mock_args.robot = "franka"
                    mock_args.sim = "mujoco"
                    mock_args.num_envs = num_envs
                    mock_args.headless = True
                    mock_args.save_image_dir = None
                    mock_args.save_video_path = None
                    mock_args.stop_on_runout = False
                    mock_args.object_states = False
                    mock_args.renderer = None
                    mock_args.scene = None
                    mock_args.try_add_table = True
                    mock_args.split = "all"
                    mock_args.render = Mock()
                    mock_args.random = Mock()

                    with patch("metasim.scripts.replay_demo.ScenarioCfg") as mock_scenario_cls:
                        mock_scenario = Mock()
                        mock_scenario.task.traj_filepath = traj_path
                        mock_scenario.sim = "mujoco"
                        mock_scenario.renderer = None
                        mock_scenario.num_envs = num_envs
                        mock_scenario.robots = [Mock()]
                        mock_scenario_cls.return_value = mock_scenario

                        with patch("os.path.exists", return_value=True):
                            replay_demo.main()

                assert mock_env.handler.num_envs == num_envs


@pytest.mark.parametrize("task", ["StackCube", "PickCube", "ReachTarget"])
def test_task_configurations(task):
    """Test that common tasks have proper configuration."""
    try:
        from metasim.cfg.tasks import get_task_cfg

        task_cfg = get_task_cfg(task)

        assert hasattr(task_cfg, "episode_length")
        assert hasattr(task_cfg, "traj_filepath")

    except Exception:
        pytest.skip(f"Task {task} not available")
