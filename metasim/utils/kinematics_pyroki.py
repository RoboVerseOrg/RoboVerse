import jax.numpy as jnp
import numpy as np
import pyroki as pk
import torch
from yourdfpy import URDF

import third_party.pyroki.examples.pyroki_snippets as pks


class get_pyroki_model:
    def __init__(self, robot_cfg):

        """Get the Pyroki robot model.

        Args:
            robot_cfg: An instance of BaseRobotCfg or similar, must contain:
                - urdf_path: str
                - ee_body_name: str (end effector link name).
        """

        self.urdf_path = robot_cfg.urdf_path
        self.ee_link_name = getattr(robot_cfg, "ee_body_name", None)
        if self.ee_link_name is None:
            raise ValueError("robot_cfg must have 'ee_body_name' defined")

        self.urdf = URDF.load(self.urdf_path)
        self.pk_robot = pk.Robot.from_urdf(self.urdf)

    def solve_ik(self, pos_target: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
        """Solve IK for a single target."""
        pos = jnp.array(pos_target.detach().cpu().numpy())
        quat = jnp.array(quat_target.detach().cpu().numpy())

        solution = pks.solve_ik(
            self.pk_robot,
            self.ee_link_name,
            target_wxyz=quat,
            target_position=pos,
        )

        q_list = np.concatenate([solution, [0.04, 0.04]])
        q_tensor = torch.tensor(q_list, dtype=torch.float32)
        return q_tensor.cuda() if torch.cuda.is_available() else q_tensor
