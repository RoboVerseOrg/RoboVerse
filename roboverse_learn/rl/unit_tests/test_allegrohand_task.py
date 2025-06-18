"""Unit tests for AllegroHand task configuration and functionality."""

import numpy as np
import pytest
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.tasks.isaacgym_envs.allegrohand_cfg import AllegroHandCfg
from metasim.utils.math import quat_inv, quat_mul
from metasim.utils.setup_util import get_robot, get_task


class TestAllegroHandTask:
    """Test suite for AllegroHand task."""

    @pytest.fixture
    def task_cfg(self):
        """Create AllegroHand task configuration."""
        return AllegroHandCfg()

    @pytest.fixture
    def mock_states(self):
        """Create mock states for testing."""
        states = [
            {
                "robots": {
                    "allegro_hand": {
                        "joint_qpos": torch.zeros(16),
                        "joint_qvel": torch.zeros(16),
                        "pos": torch.tensor([0.0, 0.0, 0.5]),
                    }
                },
                "objects": {
                    "block": {
                        "pos": torch.tensor([0.0, -0.2, 0.56]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "lin_vel": torch.zeros(3),
                        "ang_vel": torch.zeros(3),
                    },
                    "goal": {
                        "pos": torch.tensor([0.0, 0.0, 0.92]),
                        "rot": torch.tensor([0.707, 0.0, 0.707, 0.0]),  # 90 degree rotation
                    },
                },
                "extra": {
                    "actions": torch.zeros(16),
                },
            }
        ]
        return states

    @pytest.fixture
    def mock_actions(self):
        """Create mock actions for testing."""
        return [{"dof_pos_target": {f"joint_{i}": 0.1 for i in range(16)}}]

    def test_task_initialization(self, task_cfg):
        """Test that AllegroHand task initializes correctly."""
        assert task_cfg is not None
        assert task_cfg.episode_length == 600
        assert task_cfg.object_type == "block"
        assert len(task_cfg.objects) == 2  # block and goal
        assert task_cfg.objects[0].name == "block"
        assert task_cfg.objects[1].name == "goal"

    def test_observation_space(self, task_cfg):
        """Test observation space configuration."""
        obs_space = task_cfg.observation_space

        # Check robot observation space
        assert "robot" in obs_space
        assert "joint_qpos" in obs_space["robot"]
        assert obs_space["robot"]["joint_qpos"]["shape"] == (16,)

        # Check object observation space
        assert "objects" in obs_space
        assert "block" in obs_space["objects"]
        assert "goal" in obs_space["objects"]

        # Check block has position and rotation
        assert "pos" in obs_space["objects"]["block"]
        assert "rot" in obs_space["objects"]["block"]
        assert obs_space["objects"]["block"]["pos"]["shape"] == (3,)
        assert obs_space["objects"]["block"]["rot"]["shape"] == (4,)

    def test_get_observation_full_no_vel(self, task_cfg, mock_states):
        """Test get_observation method with full_no_vel type."""
        task_cfg.obs_type = "full_no_vel"
        obs = task_cfg.get_observation(mock_states)

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (1, 50)  # 1 environment, 50 dimensions

    def test_get_observation_full(self, task_cfg, mock_states):
        """Test get_observation method with full type."""
        task_cfg.obs_type = "full"
        obs = task_cfg.get_observation(mock_states)

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (1, 72)  # 1 environment, 72 dimensions

    def test_get_observation_full_state(self, task_cfg, mock_states):
        """Test get_observation method with full_state type."""
        task_cfg.obs_type = "full_state"
        obs = task_cfg.get_observation(mock_states)

        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (1, 88)  # 1 environment, 88 dimensions

    def test_reward_function(self, task_cfg, mock_states, mock_actions):
        """Test reward function computation."""
        rewards = task_cfg.reward_fn(mock_states, mock_actions)

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (1,)  # 1 environment
        assert not torch.isnan(rewards).any()
        assert not torch.isinf(rewards).any()

    def test_reward_components(self, task_cfg, mock_states, mock_actions):
        """Test individual reward components."""
        # Test with perfect alignment (goal = block pose)
        mock_states[0]["objects"]["goal"]["pos"] = mock_states[0]["objects"]["block"]["pos"].clone()
        mock_states[0]["objects"]["goal"]["rot"] = mock_states[0]["objects"]["block"]["rot"].clone()

        rewards = task_cfg.reward_fn(mock_states, mock_actions)

        # Should get success bonus when perfectly aligned
        assert rewards[0] > task_cfg.reach_goal_bonus * 0.9  # Account for action penalty

    def test_termination_function(self, task_cfg, mock_states):
        """Test termination function."""
        terminations = task_cfg.termination_fn(mock_states)

        assert isinstance(terminations, torch.Tensor)
        assert terminations.shape == (1,)  # 1 environment
        assert terminations.dtype == torch.bool

        # Should not terminate in normal conditions
        assert not terminations[0]

        # Test fall termination
        mock_states[0]["objects"]["block"]["pos"][2] = 0.1  # Very low position
        terminations = task_cfg.termination_fn(mock_states)
        assert terminations[0]  # Should terminate when object falls

    def test_quaternion_distance(self, task_cfg):
        """Test quaternion distance calculation."""
        # Identity quaternion
        q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q2 = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Test same orientation
        quat_diff = quat_mul(q1, quat_inv(q2))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[1:4], p=2), max=1.0))
        assert torch.abs(rot_dist) < 1e-6

        # Test 180 degree rotation
        q3 = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 180 degree rotation around x
        quat_diff = quat_mul(q1, quat_inv(q3))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[1:4], p=2), max=1.0))
        assert torch.abs(rot_dist - np.pi) < 1e-3

    def test_task_registration(self):
        """Test that AllegroHand task can be retrieved via get_task."""
        # Test different naming conventions
        task1 = get_task("AllegroHand")
        task2 = get_task("allegro_hand")
        task3 = get_task("isaacgym_envs:AllegroHand")
        task4 = get_task("isaacgym_envs:allegro_hand")

        assert isinstance(task1, AllegroHandCfg)
        assert isinstance(task2, AllegroHandCfg)
        assert isinstance(task3, AllegroHandCfg)
        assert isinstance(task4, AllegroHandCfg)

    def test_scenario_creation(self):
        """Test creating a scenario with AllegroHand task."""
        task = get_task("AllegroHand")
        robot = get_robot("allegro_hand")

        scenario = ScenarioCfg(task=task, robots=[robot])

        assert scenario.task is not None
        assert len(scenario.robots) == 1
        assert scenario.robots[0] is not None

    def test_reward_scale_parameters(self, task_cfg):
        """Test that reward scale parameters are set correctly."""
        assert task_cfg.dist_reward_scale == -10.0
        assert task_cfg.rot_reward_scale == 1.0
        assert task_cfg.action_penalty_scale == -0.0002
        assert task_cfg.reach_goal_bonus == 250.0
        assert task_cfg.success_tolerance == 0.1
        assert task_cfg.fall_dist == 0.24
        assert task_cfg.fall_penalty == 0.0

    def test_randomization_config(self, task_cfg):
        """Test randomization configuration."""
        assert "robot" in task_cfg.randomize
        assert "allegro_hand" in task_cfg.randomize["robot"]
        assert "joint_qpos" in task_cfg.randomize["robot"]["allegro_hand"]

        assert "object" in task_cfg.randomize
        assert "block" in task_cfg.randomize["object"]
        assert "goal" in task_cfg.randomize["object"]

        # Check block randomization ranges
        block_rand = task_cfg.randomize["object"]["block"]
        assert "position" in block_rand
        assert block_rand["position"]["x"] == [-0.01, 0.01]
        assert block_rand["position"]["y"] == [-0.01, 0.01]

        assert "orientation" in block_rand
        assert block_rand["orientation"]["x"] == [-1.0, 1.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
