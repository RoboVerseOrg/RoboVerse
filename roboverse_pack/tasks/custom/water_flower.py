"""Water flower task - Robot waters a plant using a watering can."""

from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task


@register_task("custom.water_flower", "water_flower")
class WaterFlowerTask(BaseTaskEnv):
    """Task where robot waters a flower using a watering can (sprayer)."""

    max_episode_steps = 200

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="sprayer",  # 喷壶
                scale=(1.0, 1.0, 1.0),
                physics=PhysicStateType.RIGIDBODY,
                fix_base_link=False,
                mjcf_path="/Users/AdamNg/Desktop/AxisAI/backup/axis-mvp/frontend/public/mujoco-assets/out_vhacd_h256_r1000000/sample_vhacd.xml",
            ),
            RigidObjCfg(
                name="plant_pot",  # 花盆
                scale=(1.0, 1.0, 1.0),
                physics=PhysicStateType.RIGIDBODY,
                fix_base_link=False,
                mjcf_path="/Users/AdamNg/Desktop/AxisAI/backup/axis-mvp/frontend/public/mujoco-assets/c30lumexfif4-eb_house_plant_01 (2)/eb_house_plant_01/eb_house_plant_01_pot_simple_collision.xml",
            ),
        ],
        robots=["franka"],
        simulator="mujoco",  # Using MuJoCo since we only have MJCF files
        num_envs=1,
    )

    traj_filepath = None  # Will be set via command line argument

    def _get_initial_states(self) -> list[dict] | None:
        """Return initial states based on provided initial state."""
        return [
            {
                "objects": {
                    "sprayer": {
                        "pos": torch.tensor([0.55, 0.0, 0.2]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "plant_pot": {
                        "pos": torch.tensor([0.95, 0.0, 0.2]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "franka": {
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": -0.785398,
                            "panda_joint3": 0.0,
                            "panda_joint4": -2.356194,
                            "panda_joint5": 0.0,
                            "panda_joint6": 1.570796,
                            "panda_joint7": 0.785398,
                            "panda_finger_joint1": 0.04,
                            "panda_finger_joint2": 0.04,
                        },
                    }
                },
            }
            for _ in range(self.num_envs)
        ]

    def _observation(self, states):
        """Observation space - return empty for replay tasks."""
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _reward(self, states, actions, next_states=None):
        """Reward function - not needed for replay."""
        return torch.zeros(self.num_envs, device=self.device)

    def _terminated(self, states) -> torch.Tensor:
        """Task termination - never terminate for replay."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _time_out(self, step_count) -> torch.Tensor:
        """Timeout check."""
        return step_count >= self.max_episode_steps
