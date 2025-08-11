from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.utils.state import TensorState
from scenario_cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from tasks.registry import register_task
from tasks.rl_task import RLTaskEnv


@register_task("obj_env", "obj_env", "franka.obj_env")
class ObjectEnv(RLTaskEnv):
    """Base Env for object tasks.

    This class provides common functionality for all object tasks.
    """

    def _load_task_config(self, scenario) -> None:
        """Configure common object task parameters."""
        super()._load_task_config(scenario)

        # Common configuration for object tasks
        self.objects = [
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
                urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
                mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
            ),
            ArticulationObjCfg(
                name="box_base",
                fix_base_link=True,
                usd_path="get_started/example_assets/box_base/usd/box_base.usd",
                urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
                mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
            ),
        ]
        self.reward_functions = []
        self.max_episode_steps = 100

        return scenario

    def _get_initial_states(self) -> list[dict]:
        """Get the initial states of the environment."""
        return [
            {
                "objects": {
                    "cube": {
                        "pos": torch.tensor([0.3, -0.2, 0.05]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "sphere": {
                        "pos": torch.tensor([0.4, -0.6, 0.05]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "bbq_sauce": {
                        "pos": torch.tensor([0.7, -0.3, 0.14]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                    "box_base": {
                        "pos": torch.tensor([0.5, 0.2, 0.1]),
                        "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                        "dof_pos": {"box_joint": 0.0},
                    },
                },
                "robots": {
                    "franka": {
                        "dof_pos": {
                            "panda_joint1": 0.0,
                            "panda_joint2": 0.0,
                            "panda_joint3": 0.0,
                            "panda_joint4": 0.0,
                            "panda_joint5": 0.0,
                            "panda_joint6": 0.0,
                            "panda_joint7": 0.0,
                            "panda_finger_joint1": 0.0,
                            "panda_finger_joint2": 0.0,
                        },
                        "pos": torch.tensor([0.0, 0.0, 0.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    }
                },
            }
        ]

    def _observation(self, states: TensorState) -> torch.Tensor:
        """Get the observation for the current state."""
        joint_pos = states.robots["franka"].joint_pos
        panda_hand_index = states.robots["franka"].body_names.index("panda_hand")
        ee_pos = states.robots["franka"].body_state[:, panda_hand_index, :3]

        return torch.cat([joint_pos, ee_pos], dim=1)
