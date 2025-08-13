"""Integration tests for AllegroHand environment with RL framework."""

from unittest.mock import Mock

import pytest
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.tasks.isaacgym_envs.allegrohand_cfg import AllegroHandCfg
from metasim.utils.setup_util import get_robot
from roboverse_learn.rl.env import RLEnvWrapper


class TestAllegroHandEnvironmentIntegration:
    """Test suite for AllegroHand environment integration."""

    @pytest.fixture
    def mock_sim_env(self):
        """Create a mock simulation environment."""
        mock_env = Mock()

        # Mock environment properties
        mock_env.num_envs = 4
        mock_env.device = torch.device("cpu")
        mock_env.action_space = Mock()
        mock_env.action_space.shape = (16,)
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (50,)

        # Mock reset method
        def mock_reset():
            states = {
                "robots": {
                    "allegro_hand": {
                        "joint_qpos": torch.zeros(4, 16),
                        "joint_qvel": torch.zeros(4, 16),
                        "pos": torch.zeros(4, 3),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.rand(4, 3) * 0.1 + torch.tensor([0.0, -0.2, 0.56]),
                        "rot": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
                        "lin_vel": torch.zeros(4, 3),
                        "ang_vel": torch.zeros(4, 3),
                    },
                    "goal": {
                        "pos": torch.tensor([[0.0, 0.0, 0.92]] * 4),
                        "rot": torch.rand(4, 4),
                    },
                },
                "extra": {
                    "actions": torch.zeros(4, 16),
                },
            }
            # Normalize quaternions
            for i in range(4):
                states["objects"]["goal"]["rot"][i] = torch.nn.functional.normalize(
                    states["objects"]["goal"]["rot"][i], p=2, dim=0
                )
            return states

        mock_env.reset = mock_reset
        mock_env.get_observations = Mock(return_value=torch.randn(4, 50))

        # Mock step method
        def mock_step(actions):
            rewards = torch.rand(4)
            dones = torch.zeros(4, dtype=torch.bool)
            dones[torch.rand(4) < 0.01] = True  # 1% chance of done
            timeouts = torch.zeros(4, dtype=torch.bool)
            info = {"success": torch.zeros(4, dtype=torch.bool)}
            return mock_env.get_observations(), rewards, dones, timeouts, info

        mock_env.step = mock_step
        mock_env.close = Mock()

        return mock_env

    @pytest.fixture
    def task_cfg(self):
        """Create AllegroHand task configuration."""
        return AllegroHandCfg()

    @pytest.fixture
    def scenario_cfg(self, task_cfg):
        """Create scenario configuration with AllegroHand."""
        robot = get_robot("allegro_hand")
        scenario = ScenarioCfg(task=task_cfg, robots=[robot])
        scenario.num_envs = 4
        scenario.headless = True
        return scenario

    def test_rl_env_wrapper_initialization(self, mock_sim_env):
        """Test RLEnvWrapper initialization with AllegroHand environment."""
        rl_env = RLEnvWrapper(gym_env=mock_sim_env, seed=42, verbose=False)

        assert rl_env is not None
        assert rl_env.num_envs == 4
        assert rl_env.device == torch.device("cpu")

    def test_rl_env_reset(self, mock_sim_env):
        """Test environment reset functionality."""
        rl_env = RLEnvWrapper(gym_env=mock_sim_env, seed=42, verbose=False)

        obs_dict = rl_env.reset()

        assert "obs" in obs_dict
        assert isinstance(obs_dict["obs"], torch.Tensor)
        assert obs_dict["obs"].shape == (4, 50)

    def test_rl_env_step(self, mock_sim_env):
        """Test environment step functionality."""
        rl_env = RLEnvWrapper(gym_env=mock_sim_env, seed=42, verbose=False)

        # Reset environment
        rl_env.reset()

        # Take a random action
        actions = torch.randn(4, 16)
        obs_dict, rewards, dones, timeouts, info = rl_env.step(actions)

        assert "obs" in obs_dict
        assert isinstance(obs_dict["obs"], torch.Tensor)
        assert obs_dict["obs"].shape == (4, 50)

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (4,)

        assert isinstance(dones, torch.Tensor)
        assert dones.shape == (4,)
        assert dones.dtype == torch.bool

        assert isinstance(info, dict)

    def test_action_space_compatibility(self, task_cfg):
        """Test that action space matches AllegroHand DOF."""
        # AllegroHand has 16 DOF
        expected_action_dim = 16

        # Check task configuration
        assert hasattr(task_cfg, "observation_space")
        robot_obs = task_cfg.observation_space["robot"]
        assert robot_obs["joint_qpos"]["shape"] == (16,)

    def test_observation_space_compatibility(self, task_cfg):
        """Test observation space dimensions for different observation types."""
        # Test full_no_vel observation
        task_cfg.obs_type = "full_no_vel"
        mock_states = [
            {
                "robots": {
                    "allegro_hand": {
                        "joint_qpos": torch.zeros(16),
                        "pos": torch.zeros(3),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.zeros(3),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "goal": {
                        "pos": torch.zeros(3),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "extra": {
                    "actions": torch.zeros(16),
                },
            }
        ]

        obs = task_cfg.get_observation(mock_states)
        assert obs.shape == (1, 50)

    def test_reward_computation_integration(self, task_cfg):
        """Test reward computation with various state configurations."""
        # Test states with varying distances and orientations
        test_cases = [
            # Perfect alignment - should get high reward
            {
                "block_pos": torch.tensor([0.0, 0.0, 0.6]),
                "block_rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "goal_pos": torch.tensor([0.0, 0.0, 0.6]),
                "goal_rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "expected_reward_min": 200.0,  # Should get success bonus
            },
            # Far from goal - should get negative reward
            {
                "block_pos": torch.tensor([1.0, 1.0, 0.6]),
                "block_rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "goal_pos": torch.tensor([0.0, 0.0, 0.6]),
                "goal_rot": torch.tensor([0.0, 0.707, 0.707, 0.0]),
                "expected_reward_max": 0.0,  # Should be negative
            },
        ]

        for test_case in test_cases:
            states = [
                {
                    "robots": {
                        "allegro_hand": {
                            "pos": torch.tensor([0.0, 0.0, 0.5]),
                        }
                    },
                    "objects": {
                        "block": {
                            "pos": test_case["block_pos"],
                            "rot": test_case["block_rot"],
                        },
                        "goal": {
                            "pos": test_case["goal_pos"],
                            "rot": test_case["goal_rot"],
                        },
                    },
                }
            ]

            actions = [{"dof_pos_target": {f"joint_{i}": 0.0 for i in range(16)}}]

            rewards = task_cfg.reward_fn(states, actions)

            if "expected_reward_min" in test_case:
                assert rewards[0] >= test_case["expected_reward_min"]
            if "expected_reward_max" in test_case:
                assert rewards[0] <= test_case["expected_reward_max"]

    def test_termination_conditions(self, task_cfg):
        """Test various termination conditions."""
        # Test normal condition - should not terminate
        states_normal = [
            {
                "robots": {
                    "allegro_hand": {
                        "pos": torch.tensor([0.0, 0.0, 0.5]),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.tensor([0.0, -0.2, 0.56]),
                    }
                },
            }
        ]

        terminations = task_cfg.termination_fn(states_normal)
        assert not terminations[0]

        # Test fall condition - should terminate
        states_fall = [
            {
                "robots": {
                    "allegro_hand": {
                        "pos": torch.tensor([0.0, 0.0, 0.5]),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.tensor([0.0, -0.2, 0.1]),  # Very low position
                    }
                },
            }
        ]

        terminations = task_cfg.termination_fn(states_fall)
        assert terminations[0]

    def test_multi_env_batch_processing(self, task_cfg):
        """Test that task handles multiple environments correctly."""
        num_envs = 8

        # Create batch of states
        states = []
        for i in range(num_envs):
            states.append({
                "robots": {
                    "allegro_hand": {
                        "joint_qpos": torch.randn(16),
                        "joint_qvel": torch.randn(16),
                        "pos": torch.tensor([0.0, 0.0, 0.5]),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.randn(3) * 0.1 + torch.tensor([0.0, -0.2, 0.56]),
                        "rot": torch.nn.functional.normalize(torch.randn(4), p=2, dim=0),
                        "lin_vel": torch.randn(3) * 0.1,
                        "ang_vel": torch.randn(3) * 0.1,
                    },
                    "goal": {
                        "pos": torch.tensor([0.0, 0.0, 0.92]),
                        "rot": torch.nn.functional.normalize(torch.randn(4), p=2, dim=0),
                    },
                },
                "extra": {
                    "actions": torch.randn(16) * 0.1,
                },
            })

        # Test observation computation
        obs = task_cfg.get_observation(states)
        assert obs.shape == (num_envs, 50)

        # Test reward computation
        actions = []
        for i in range(num_envs):
            actions.append({"dof_pos_target": {f"joint_{j}": torch.randn(1).item() * 0.1 for j in range(16)}})

        rewards = task_cfg.reward_fn(states, actions)
        assert rewards.shape == (num_envs,)

        # Test termination computation
        terminations = task_cfg.termination_fn(states)
        assert terminations.shape == (num_envs,)
        assert terminations.dtype == torch.bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
