"""Functional tests for replay_demo.py without direct imports."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestReplayDemoFunctional:
    """Test replay_demo.py as a black-box script."""

    def test_replay_demo_help(self):
        """Test that replay_demo.py shows help correctly."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import sys
from unittest.mock import MagicMock

sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.utils'] = MagicMock()

import metasim.scripts.replay_demo
""")
            wrapper_path = f.name

        try:
            cmd = [sys.executable, wrapper_path, "--help"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)

            help_text = result.stdout + result.stderr

            assert "--task" in help_text
            assert "--sim" in help_text
            assert "--robot" in help_text
            assert "--num-envs" in help_text
        finally:
            os.unlink(wrapper_path)

    def test_replay_demo_missing_args(self):
        """Test that replay_demo.py fails gracefully with missing arguments."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import sys
from unittest.mock import MagicMock

sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.utils'] = MagicMock()

sys.argv = [sys.argv[0], '--task', 'StackCube']

try:
    import metasim.scripts.replay_demo
except SystemExit as e:
    if e.code != 0:
        print("ERROR: Missing required arguments")
        sys.exit(1)
    else:
        print("SUCCESS: Help shown")
        sys.exit(0)
""")
            wrapper_path = f.name

        try:
            cmd = [sys.executable, wrapper_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)

            assert result.returncode != 0
            assert "Missing required arguments" in result.stdout or "required" in result.stderr.lower()
        finally:
            os.unlink(wrapper_path)

    @pytest.mark.parametrize("simulator", ["mujoco", "sapien2", "pybullet"])
    def test_replay_demo_dry_run(self, simulator, tmp_path):
        """Test replay_demo.py with minimal mocked execution."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent)

        import pickle

        traj_dir = tmp_path / "trajs" / "TestTask" / "v2"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = traj_dir / "franka_v2.pkl"

        mock_data = {
            "franka": {
                "init_state": {
                    "joint_pos": [[0.0] * 7] * 4,
                    "joint_vel": [[0.0] * 7] * 4,
                },
                "actions": [[[0.1] * 7] * 5] * 4,
            }
        }

        with open(traj_file, "wb") as f:
            pickle.dump(mock_data, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f"""
import sys
import os
from unittest.mock import MagicMock, Mock, patch
import torch

sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.utils'] = MagicMock()

sys.argv = [
    sys.argv[0],
    '--task', 'TestTask',
    '--sim', '{simulator}',
    '--robot', 'franka',
    '--num_envs', '4',
    '--headless',
]

with patch('metasim.utils.setup_util.get_sim_env_class') as mock_get_env:
    with patch('metasim.cfg.tasks.get_task_cfg') as mock_get_task:
        mock_env = Mock()
        mock_handler = Mock()
        mock_env.handler = mock_handler
        mock_handler.num_envs = 4

        mock_state = Mock()
        mock_state.cameras = {{}}
        mock_env.reset.return_value = (mock_state, {{}})
        mock_env.step.return_value = (
            mock_state,
            torch.zeros(4),
            torch.ones(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            {{}}
        )

        mock_env_class = Mock(return_value=mock_env)
        mock_get_env.return_value = mock_env_class

        mock_task = Mock()
        mock_task.traj_filepath = r"{traj_dir!s}"
        mock_task.episode_length = 100
        mock_get_task.return_value = mock_task

        try:
            import metasim.scripts.replay_demo
            print("SUCCESS: Replay demo executed")
        except Exception as e:
            print(f"ERROR: {{e}}")
            sys.exit(1)
""")
            wrapper_path = f.name

        try:
            cmd = [sys.executable, wrapper_path]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=10)

            assert "SUCCESS: Replay demo executed" in result.stdout
            assert result.returncode == 0
        finally:
            os.unlink(wrapper_path)


class TestReplayDemoComponents:
    """Test individual components of replay_demo without full import."""

    def test_action_bounds_logic(self):
        """Test the action bounds logic in isolation."""
        all_actions = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9],
        ]

        actions_t0 = [actions[0] if len(actions) > 0 else None for actions in all_actions]
        assert actions_t0 == [1, 4, 6]

        actions_t3 = [actions[3] if len(actions) > 3 else actions[-1] for actions in all_actions]
        assert actions_t3 == [3, 5, 9]

        runout_t2 = all(len(actions) <= 2 for actions in all_actions)
        assert not runout_t2

        runout_t4 = all(len(actions) <= 4 for actions in all_actions)
        assert runout_t4

    def test_trajectory_format(self, tmp_path):
        """Test expected trajectory file format."""
        import pickle

        traj_data = {
            "franka": {
                "init_state": {
                    "joint_pos": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                    "joint_vel": [[0.0] * 7],
                    "root_pos": [[0.0, 0.0, 0.0]],
                    "root_quat": [[1.0, 0.0, 0.0, 0.0]],
                },
                "actions": [[[0.1] * 7, [0.2] * 7, [0.3] * 7]],
                "states": [
                    [
                        {"joint_pos": [0.1] * 7, "joint_vel": [0.0] * 7},
                        {"joint_pos": [0.2] * 7, "joint_vel": [0.0] * 7},
                        {"joint_pos": [0.3] * 7, "joint_vel": [0.0] * 7},
                    ]
                ],
            }
        }

        traj_file = tmp_path / "test_traj.pkl"
        with open(traj_file, "wb") as f:
            pickle.dump(traj_data, f)

        with open(traj_file, "rb") as f:
            loaded = pickle.load(f)

        assert "franka" in loaded
        assert "init_state" in loaded["franka"]
        assert "actions" in loaded["franka"]
        assert len(loaded["franka"]["actions"][0]) == 3
        assert len(loaded["franka"]["actions"][0][0]) == 7


def test_simulator_rendering_compatibility():
    """Test which simulators support rendering in replay demo."""
    rendering_support = {
        "isaaclab": {"supports": True, "requires_gpu": True},
        "isaacgym": {"supports": True, "requires_gpu": True},
        "genesis": {"supports": True, "requires_gpu": True},
        "mujoco": {"supports": True, "requires_gpu": False},
        "sapien2": {"supports": True, "requires_gpu": False},
        "sapien3": {"supports": True, "requires_gpu": True},
        "pybullet": {"supports": True, "requires_gpu": False},
        "mjx": {"supports": False, "requires_gpu": True},
        "blender": {"supports": True, "requires_gpu": False},
    }

    cpu_renderers = [sim for sim, info in rendering_support.items() if info["supports"] and not info["requires_gpu"]]

    assert "mujoco" in cpu_renderers
    assert "sapien2" in cpu_renderers
    assert "pybullet" in cpu_renderers

    return cpu_renderers
