from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.state import TensorState


@register_task("xhand_env", "xhand.xhand_env")
class XHandEnv(BaseTaskEnv):
    """Environment for XHand-style object tasks (robot-agnostic).

    Notes:
    -----
    * Initial states provide BOTH:
        - state["robots"][<robot_name>] = {...}
        - state[<robot_name>] = {...}
      This makes it robust across different simulator state readers.
    * Observation is robot-agnostic and chooses an EE link by name heuristic.
    """

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=[0.1, 0.1, 0.1],
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
            ),
        ],
        robots=[
            RobotCfg(
                name="xhand_right",
                # usd_path="roboverse_data/robots/xhand_right/usd/xhand_right.usd",
                urdf_path="roboverse_data/robots/xhand_right/urdf/xhand_right.urdf",
                # mjcf_path="roboverse_data/robots/xhand_right/mjcf/xhand_right.xml",
                # mjx_mjcf_path="roboverse_data/robots/xhand_right/mjx/xhand_right.xml",
            ),
        ],
    )

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)
        self.reward_functions = []
        self.max_episode_steps = 100

    # --------------------------------------------------------------------- #
    # Initial states                                                        #
    # --------------------------------------------------------------------- #
    def _discover_robot_names(self) -> list[str]:
        """Try to discover robot names from the task/scenario."""
        names: list[str] = []
        if hasattr(self, "robot_names") and self.robot_names:
            names.extend(list(self.robot_names))
        elif hasattr(self, "robots"):
            try:
                names.extend(list(self.robots.keys()))
            except Exception:
                pass
        # Robust fallbacks in case discovery fails or differs by build
        for n in ["xhand_right", "xhand", "franka", "robot"]:
            if n not in names:
                names.append(n)
        return names

    def _get_initial_states(self) -> list[dict]:
        """Return initial states that include BOTH 'robots' and top-level robot keys.

        This satisfies readers that expect:
          - state["robots"][robot_name]["pos"], and/or
          - state[robot_name]["pos"]
        """
        states: list[dict] = []
        for _ in range(self.num_envs):
            names = self._discover_robot_names()

            # One canonical robot state block referenced by every alias
            robot_state_block = {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
                # Optionally: "dof_pos": { "<joint_name>": 0.0, ... }
            }

            robots_dict = {n: robot_state_block for n in names}

            init = {
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
                # Path 1: readers that expect state["robots"][name]
                "robots": robots_dict,
                # Path 2: readers that expect state[name] at top level
                **{n: robot_state_block for n in names},
            }
            states.append(init)
        return states

    # --------------------------------------------------------------------- #
    # Observation                                                           #
    # --------------------------------------------------------------------- #
    def _observation(self, states: TensorState) -> torch.Tensor:
        """Robot-agnostic observation: [joint_pos, ee_pos].

        The end-effector body is chosen by looking for names containing 'palm'
        or 'hand'; if none found, falls back to the first body.
        """
        # Use the first available robot in the state container
        robot_name = next(iter(states.robots.keys()))
        robot_state = states.robots[robot_name]

        joint_pos = robot_state.joint_pos

        # Heuristically pick an EE body
        body_names = robot_state.body_names
        ee_candidates = [i for i, n in enumerate(body_names) if ("palm" in n.lower()) or ("hand" in n.lower())]
        ee_index = ee_candidates[0] if ee_candidates else 0
        ee_pos = robot_state.body_state[:, ee_index, :3]

        return torch.cat([joint_pos, ee_pos], dim=1)
