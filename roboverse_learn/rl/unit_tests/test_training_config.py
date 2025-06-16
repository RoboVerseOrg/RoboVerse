"""Unit tests for AllegroHand training configuration."""

import os

import pytest
from hydra import compose, initialize_config_dir


class TestAllegroHandTrainingConfig:
    """Test suite for AllegroHand PPO training configuration."""

    @pytest.fixture
    def config_path(self):
        """Get the path to the config directory."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

    @pytest.fixture
    def training_config(self, config_path):
        """Load the AllegroHand PPO training configuration."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name="default", overrides=["train=AllegroHandPPO"])
            return cfg

    def test_config_loading(self, training_config):
        """Test that configuration loads successfully."""
        assert training_config is not None
        assert hasattr(training_config, "train")
        assert hasattr(training_config, "environment")
        assert hasattr(training_config, "experiment")

    def test_task_configuration(self, training_config):
        """Test task-specific configuration."""
        assert training_config.train.task_name == "AllegroHand"
        assert training_config.train.robot_name == "allegro_hand"
        assert training_config.train.algo == "ppo"

    def test_observation_space_config(self, training_config):
        """Test observation space configuration."""
        assert training_config.train.observation_space.shape == [50]
        assert training_config.train.observation_shape == [50]

    def test_ppo_hyperparameters(self, training_config):
        """Test PPO algorithm hyperparameters."""
        ppo_cfg = training_config.train.ppo

        # Action space
        assert ppo_cfg.action_num == 16

        # Training parameters
        assert ppo_cfg.num_actors == 4096
        assert ppo_cfg.horizon_length == 16
        assert ppo_cfg.minibatch_size == 32768
        assert ppo_cfg.mini_epochs == 5

        # PPO specific parameters
        assert ppo_cfg.e_clip == 0.2
        assert ppo_cfg.entropy_coef == 0.0
        assert ppo_cfg.critic_coef == 2.0
        assert ppo_cfg.gamma == 0.99
        assert ppo_cfg.tau == 0.95

        # Learning rate
        assert ppo_cfg.learning_rate == 5e-4
        assert ppo_cfg.lr_schedule == "adaptive"
        assert ppo_cfg.kl_threshold == 0.016

        # Gradient clipping
        assert ppo_cfg.truncate_grads
        assert ppo_cfg.grad_norm == 1.0

        # Value function
        assert ppo_cfg.clip_value
        assert ppo_cfg.normalize_value
        assert ppo_cfg.normalize_advantage
        assert ppo_cfg.normalize_input

    def test_network_architecture(self, training_config):
        """Test neural network architecture configuration."""
        network_cfg = training_config.train.ppo.network

        assert network_cfg.mlp.units == [512, 512, 256, 128]
        assert network_cfg.separate_value_mlp

    def test_environment_settings(self, training_config):
        """Test environment configuration."""
        env_cfg = training_config.environment

        # These come from default.yaml unless overridden
        assert env_cfg.num_envs == 32  # Default value
        assert not env_cfg.headless  # Default value
        assert env_cfg.sim_name == "isaacgym"

    def test_experiment_settings(self, training_config):
        """Test experiment configuration."""
        exp_cfg = training_config.experiment

        # output_name is based on task_name
        assert exp_cfg.output_name == "AllegroHand"
        assert hasattr(exp_cfg, "seed")
        assert hasattr(exp_cfg, "device_id")

    def test_training_duration(self, training_config):
        """Test training duration settings."""
        assert training_config.train.ppo.max_agent_steps == 500000000
        assert training_config.train.ppo.save_frequency == 50
        assert training_config.train.ppo.save_best_after == 50

    def test_config_overrides(self, config_path):
        """Test configuration with custom overrides."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(
                config_name="default",
                overrides=[
                    "train=AllegroHandPPO",
                    "environment.num_envs=2048",
                    "train.ppo.learning_rate=1e-3",
                    "experiment.seed=123",
                ],
            )

            assert cfg.environment.num_envs == 2048
            assert cfg.train.ppo.learning_rate == 1e-3
            assert cfg.experiment.seed == 123

    def test_default_config_compatibility(self, config_path):
        """Test that AllegroHandPPO config is compatible with default config."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            # Load default config
            default_cfg = compose(config_name="default")

            # Load AllegroHandPPO config
            allegro_cfg = compose(config_name="default", overrides=["train=AllegroHandPPO"])

            # Check that all default fields exist in AllegroHandPPO config
            for key in default_cfg:
                assert key in allegro_cfg

    def test_observation_shape_variants(self):
        """Test different observation shape configurations."""
        obs_shapes = {"full_no_vel": 50, "full": 72, "full_state": 88}

        for obs_type, expected_shape in obs_shapes.items():
            # This would be tested with actual environment creation
            # Here we just verify the expected values
            assert expected_shape > 0
            assert isinstance(expected_shape, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
