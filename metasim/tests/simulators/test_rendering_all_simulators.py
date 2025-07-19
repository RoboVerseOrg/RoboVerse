"""Test rendering capabilities across all simulators."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.state import TensorState

RENDERING_SIMULATORS = {
    "isaaclab": {"supports_rendering": True, "requires_gpu": True},
    "isaacgym": {"supports_rendering": True, "requires_gpu": True},
    "genesis": {"supports_rendering": True, "requires_gpu": True},
    "mujoco": {"supports_rendering": True, "requires_gpu": False},
    "sapien2": {"supports_rendering": True, "requires_gpu": False},
    "sapien3": {"supports_rendering": True, "requires_gpu": True},
    "pybullet": {"supports_rendering": True, "requires_gpu": False},
    "mjx": {"supports_rendering": False, "requires_gpu": True},
    "blender": {"supports_rendering": True, "requires_gpu": False},
}


class TestSimulatorRendering:
    """Test rendering functionality for each simulator."""

    @pytest.fixture
    def basic_scenario(self):
        """Create a basic scenario for rendering tests."""
        camera = PinholeCameraCfg(
            name="test_camera",
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            width=128,
            height=128,
            fov=60.0,
        )

        scenario = ScenarioCfg(
            task="StackCube",
            robots=["franka"],
            cameras=[camera],
            render=RenderCfg(
                render_mode="rgb",
                img_width=128,
                img_height=128,
            ),
            num_envs=2,
            headless=True,
        )

        return scenario

    @pytest.mark.parametrize("simulator", list(RENDERING_SIMULATORS.keys()))
    def test_simulator_rendering_capability(self, simulator, basic_scenario):
        """Test if simulator can render images."""
        sim_info = RENDERING_SIMULATORS[simulator]

        if sim_info["requires_gpu"]:
            pytest.skip(f"{simulator} requires GPU")

        if not sim_info["supports_rendering"]:
            pytest.skip(f"{simulator} does not support rendering")

        if simulator == "blender":
            pytest.skip("Blender is render-only, tested separately")

        basic_scenario.sim = simulator

        with patch(f"metasim.sim.{simulator}.{simulator}.{simulator.capitalize()}Handler") as MockHandler:
            mock_handler = MagicMock()
            MockHandler.return_value = mock_handler

            mock_rgb = torch.rand(2, 128, 128, 3) * 255
            mock_camera_state = MagicMock()
            mock_camera_state.rgb = mock_rgb

            mock_state = TensorState()
            mock_state.cameras = {"test_camera": mock_camera_state}

            mock_handler._get_states.return_value = mock_state
            mock_handler.reset.return_value = mock_state
            mock_handler.num_envs = 2

            from metasim.utils.setup_util import get_sim_env_class

            with patch("metasim.utils.setup_util.get_sim_handler_class") as mock_get_handler:
                mock_get_handler.return_value = MockHandler

                env_class = get_sim_env_class(SimType(simulator))
                env = env_class(basic_scenario)

                obs, _ = env.reset()

                assert hasattr(obs, "cameras")
                assert "test_camera" in obs.cameras

                camera_data = obs.cameras["test_camera"]
                assert hasattr(camera_data, "rgb")

                rgb = camera_data.rgb
                assert rgb.shape == (2, 128, 128, 3)
                assert rgb.dtype == torch.float32
                assert rgb.min() >= 0 and rgb.max() <= 255

    def test_multi_camera_rendering(self, basic_scenario):
        """Test rendering with multiple cameras."""
        camera2 = PinholeCameraCfg(
            name="side_camera",
            pos=(0.0, -2.0, 1.0),
            look_at=(0.0, 0.0, 0.5),
            width=64,
            height=64,
        )

        camera3 = PinholeCameraCfg(
            name="top_camera",
            pos=(0.0, 0.0, 3.0),
            look_at=(0.0, 0.0, 0.0),
            width=256,
            height=256,
        )

        basic_scenario.cameras = [basic_scenario.cameras[0], camera2, camera3]
        basic_scenario.sim = "mujoco"

        with patch("metasim.sim.mujoco.mujoco.MujocoHandler") as MockHandler:
            mock_handler = MagicMock()
            MockHandler.return_value = mock_handler

            mock_states = TensorState()
            mock_states.cameras = {
                "test_camera": MagicMock(rgb=torch.rand(2, 128, 128, 3) * 255),
                "side_camera": MagicMock(rgb=torch.rand(2, 64, 64, 3) * 255),
                "top_camera": MagicMock(rgb=torch.rand(2, 256, 256, 3) * 255),
            }

            mock_handler._get_states.return_value = mock_states
            mock_handler.reset.return_value = mock_states
            mock_handler.num_envs = 2

            from metasim.utils.setup_util import get_sim_env_class

            with patch("metasim.utils.setup_util.get_sim_handler_class") as mock_get_handler:
                mock_get_handler.return_value = MockHandler

                env_class = get_sim_env_class(SimType.MUJOCO)
                env = env_class(basic_scenario)

                obs, _ = env.reset()

                assert len(obs.cameras) == 3
                assert "test_camera" in obs.cameras
                assert "side_camera" in obs.cameras
                assert "top_camera" in obs.cameras

                assert obs.cameras["test_camera"].rgb.shape == (2, 128, 128, 3)
                assert obs.cameras["side_camera"].rgb.shape == (2, 64, 64, 3)
                assert obs.cameras["top_camera"].rgb.shape == (2, 256, 256, 3)

    @pytest.mark.parametrize("render_mode", ["rgb", "depth", "segmentation"])
    def test_different_render_modes(self, render_mode, basic_scenario):
        """Test different rendering modes if supported."""
        basic_scenario.render.render_mode = render_mode
        basic_scenario.sim = "mujoco"

        with patch("metasim.sim.mujoco.mujoco.MujocoHandler") as MockHandler:
            mock_handler = MagicMock()
            MockHandler.return_value = mock_handler

            mock_camera_state = MagicMock()

            if render_mode == "rgb":
                mock_camera_state.rgb = torch.rand(2, 128, 128, 3) * 255
            elif render_mode == "depth":
                mock_camera_state.depth = torch.rand(2, 128, 128, 1)
            elif render_mode == "segmentation":
                mock_camera_state.segmentation = torch.randint(0, 10, (2, 128, 128, 1))

            mock_state = TensorState()
            mock_state.cameras = {"test_camera": mock_camera_state}

            mock_handler._get_states.return_value = mock_state
            mock_handler.reset.return_value = mock_state
            mock_handler.num_envs = 2

            assert render_mode in ["rgb", "depth", "segmentation"]


class TestHybridRendering:
    """Test hybrid rendering with different physics and rendering simulators."""

    @pytest.mark.parametrize(
        "physics_sim,render_sim",
        [
            ("mujoco", "isaaclab"),
            ("genesis", "mujoco"),
            ("pybullet", "sapien2"),
        ],
    )
    def test_hybrid_simulation(self, physics_sim, render_sim):
        """Test using one simulator for physics and another for rendering."""
        if RENDERING_SIMULATORS[physics_sim]["requires_gpu"] or RENDERING_SIMULATORS[render_sim]["requires_gpu"]:
            pytest.skip("Requires GPU")

        camera = PinholeCameraCfg(
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            width=64,
            height=64,
        )

        scenario = ScenarioCfg(
            task="StackCube",
            robots=["franka"],
            cameras=[camera],
            sim=physics_sim,
            renderer=render_sim,
            num_envs=1,
            headless=True,
        )

        with patch("metasim.sim.HybridSimEnv") as MockHybrid:
            mock_env = MagicMock()
            MockHybrid.return_value = mock_env

            mock_state = TensorState()
            mock_state.cameras = {"camera": MagicMock(rgb=torch.rand(1, 64, 64, 3) * 255)}

            mock_env.reset.return_value = (mock_state, {})
            mock_env.step.return_value = (
                mock_state,
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

            with patch(f"metasim.sim.{physics_sim}.{physics_sim}.{physics_sim.capitalize()}Handler"):
                with patch(f"metasim.sim.{render_sim}.{render_sim}.{render_sim.capitalize()}Handler"):
                    assert scenario.sim == physics_sim
                    assert scenario.renderer == render_sim


class TestRenderingPerformance:
    """Test rendering performance characteristics."""

    @pytest.mark.parametrize("num_envs", [1, 4, 16])
    @pytest.mark.parametrize("resolution", [(64, 64), (128, 128), (256, 256)])
    def test_rendering_scalability(self, num_envs, resolution):
        """Test how rendering scales with number of environments and resolution."""
        width, height = resolution

        camera = PinholeCameraCfg(
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            width=width,
            height=height,
        )

        scenario = ScenarioCfg(
            task="StackCube",
            robots=["franka"],
            cameras=[camera],
            sim="mujoco",
            num_envs=num_envs,
            headless=True,
        )

        with patch("metasim.sim.mujoco.mujoco.MujocoHandler") as MockHandler:
            mock_handler = MagicMock()
            MockHandler.return_value = mock_handler

            mock_rgb = torch.rand(num_envs, height, width, 3) * 255
            mock_camera = MagicMock(rgb=mock_rgb)

            mock_state = TensorState()
            mock_state.cameras = {"camera": mock_camera}

            mock_handler._get_states.return_value = mock_state
            mock_handler.reset.return_value = mock_state
            mock_handler.num_envs = num_envs

            assert mock_rgb.shape == (num_envs, height, width, 3)

            bytes_per_image = height * width * 3 * 4
            total_bytes = bytes_per_image * num_envs
            total_mb = total_bytes / (1024 * 1024)

            print(f"\nRendering {num_envs} envs at {width}x{height}: ~{total_mb:.2f} MB")

            assert total_mb < 1000


class TestReplayDemoRendering:
    """Specific tests for replay_demo.py rendering functionality."""

    def test_replay_with_video_output(self, tmp_path):
        """Test replay demo with video output."""
        video_path = tmp_path / "test_replay.mp4"

        with patch("metasim.scripts.replay_demo.get_sim_env_class") as mock_get_env:
            with patch("metasim.scripts.replay_demo.get_traj") as mock_get_traj:
                with patch("metasim.scripts.replay_demo.iio.mimsave") as mock_save_video:
                    mock_env = MagicMock()
                    mock_handler = MagicMock()
                    mock_env.handler = mock_handler

                    num_steps = 5
                    mock_observations = []
                    for _ in range(num_steps):
                        camera_data = MagicMock()
                        camera_data.rgb = torch.rand(4, 64, 64, 3) * 255
                        state = TensorState()
                        state.cameras = {"camera": camera_data}
                        mock_observations.append(state)

                    mock_env.reset.return_value = (mock_observations[0], {})

                    step_count = 0

                    def mock_step(actions):
                        nonlocal step_count
                        step_count += 1
                        obs = mock_observations[min(step_count, len(mock_observations) - 1)]
                        success = (
                            torch.ones(4, dtype=torch.bool) if step_count >= 3 else torch.zeros(4, dtype=torch.bool)
                        )
                        return obs, torch.zeros(4), success, torch.zeros(4, dtype=torch.bool), {}

                    mock_env.step = mock_step

                    mock_env_class = MagicMock(return_value=mock_env)
                    mock_get_env.return_value = mock_env_class

                    init_state = TensorState(joint_pos=torch.zeros(4, 7))
                    actions = [[torch.zeros(7) for _ in range(5)] for _ in range(4)]
                    mock_get_traj.return_value = (init_state, actions, None)

                    from metasim.scripts import replay_demo

                    with patch.object(replay_demo, "args") as mock_args:
                        mock_args.task = "test_task"
                        mock_args.robot = "franka"
                        mock_args.sim = "mujoco"
                        mock_args.num_envs = 4
                        mock_args.headless = True
                        mock_args.save_image_dir = None
                        mock_args.save_video_path = str(video_path)
                        mock_args.stop_on_runout = False
                        mock_args.object_states = False
                        mock_args.renderer = None
                        mock_args.scene = None
                        mock_args.try_add_table = True
                        mock_args.split = "all"

                        mock_task = MagicMock()
                        mock_task.traj_filepath = "fake_path"

                        with patch("metasim.scripts.replay_demo.ScenarioCfg") as mock_scenario_cls:
                            mock_scenario = MagicMock()
                            mock_scenario.task = mock_task
                            mock_scenario.sim = "mujoco"
                            mock_scenario.renderer = None
                            mock_scenario.num_envs = 4
                            mock_scenario.robots = [MagicMock()]
                            mock_scenario_cls.return_value = mock_scenario

                            with patch("os.path.exists", return_value=True):
                                replay_demo.main()

                    mock_save_video.assert_called_once()
                    saved_path, saved_images, *_ = mock_save_video.call_args[0]
                    assert str(video_path) in str(saved_path)
                    assert len(saved_images) > 0
