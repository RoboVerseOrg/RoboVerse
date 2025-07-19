"""Test replay_demo.py with torchvision mocked to avoid version conflicts."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.utils"] = MagicMock()


class TestReplayDemoWithMockedDeps:
    """Test replay demo functionality with mocked dependencies."""

    def test_replay_demo_core_functions(self):
        """Test core functions of replay_demo.py."""
        from metasim.cfg.robots.franka_cfg import FrankaCfg
        from metasim.scripts.replay_demo import get_actions, get_runout, get_states
        from metasim.utils.state import TensorState

        robot = FrankaCfg()
        all_actions = [
            [torch.tensor([1.0] * 7), torch.tensor([2.0] * 7)],
            [torch.tensor([3.0] * 7), torch.tensor([4.0] * 7)],
        ]

        actions = get_actions(all_actions, 0, 2, robot)
        assert len(actions) == 2
        assert torch.allclose(actions[0], torch.tensor([1.0] * 7))
        assert torch.allclose(actions[1], torch.tensor([3.0] * 7))

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

        test_actions = [
            [1, 2, 3],
            [1, 2],
            [1, 2, 3, 4],
        ]

        assert not get_runout(test_actions, 1)
        assert get_runout(test_actions, 4)

    def test_obs_saver_without_io(self):
        """Test ObsSaver logic without actual file I/O."""
        from metasim.scripts.replay_demo import ObsSaver
        from metasim.utils.state import TensorState

        saver = ObsSaver()
        assert saver.video_path is None
        assert saver.image_dir is None
        assert len(saver.images) == 0

        camera_data = MagicMock()
        camera_data.rgb = torch.rand(4, 64, 64, 3) * 255

        state = TensorState()
        state.cameras = {"camera": camera_data}

        saver.add(state)
        saver.add(state)

        assert len(saver.images) == 2

    @pytest.mark.parametrize("simulator", ["mujoco", "sapien2", "pybullet"])
    def test_replay_demo_workflow(self, simulator, tmp_path):
        """Test the replay demo workflow with fully mocked components."""
        import pickle

        from metasim.scripts import replay_demo
        from metasim.utils.state import TensorState

        traj_dir = tmp_path / "trajs" / "test_task" / "v2"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = traj_dir / "franka_v2.pkl"

        mock_data = {
            "franka": {
                "init_state": {
                    "joint_pos": np.random.randn(4, 7).tolist(),
                    "joint_vel": np.zeros((4, 7)).tolist(),
                },
                "actions": [[np.random.randn(7).tolist() for _ in range(5)] for _ in range(4)],
            }
        }

        with open(traj_file, "wb") as f:
            pickle.dump(mock_data, f)

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env_class:
            with patch("metasim.scripts.replay_demo.get_traj") as mock_get_traj:
                with patch("metasim.scripts.replay_demo.ScenarioCfg") as mock_scenario_cls:
                    with patch.object(replay_demo, "args") as mock_args:
                        mock_env = MagicMock()
                        mock_handler = MagicMock()
                        mock_env.handler = mock_handler
                        mock_handler.num_envs = 4

                        init_state = TensorState(joint_pos=torch.zeros(4, 7))
                        obs_state = TensorState()
                        obs_state.cameras = {}

                        mock_env.reset.return_value = (obs_state, {})
                        mock_env.step.return_value = (
                            obs_state,
                            torch.zeros(4),
                            torch.ones(4, dtype=torch.bool),
                            torch.zeros(4, dtype=torch.bool),
                            {},
                        )

                        mock_get_env_class.return_value = Mock(return_value=mock_env)

                        actions = [[torch.zeros(7)] * 5 for _ in range(4)]
                        mock_get_traj.return_value = (init_state, actions, None)

                        mock_args.task = "test_task"
                        mock_args.robot = "franka"
                        mock_args.sim = simulator
                        mock_args.num_envs = 4
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

                        mock_scenario = Mock()
                        mock_scenario.task.traj_filepath = str(traj_dir)
                        mock_scenario.sim = simulator
                        mock_scenario.renderer = None
                        mock_scenario.num_envs = 4
                        mock_scenario.robots = [Mock()]
                        mock_scenario_cls.return_value = mock_scenario

                        with patch("os.path.exists", return_value=True):
                            replay_demo.main()

                        assert mock_env.reset.called
                        assert mock_env.step.called
                        assert mock_env.close.called


def test_replay_demo_entry_points():
    """Test that replay_demo has the expected entry points."""
    from metasim.scripts import replay_demo

    assert hasattr(replay_demo, "main")
    assert callable(replay_demo.main)

    assert hasattr(replay_demo, "get_actions")
    assert hasattr(replay_demo, "get_states")
    assert hasattr(replay_demo, "get_runout")
    assert hasattr(replay_demo, "ObsSaver")

    assert isinstance(replay_demo.ObsSaver, type)
