import torch
import numpy as np
import pyroki as pk
from yourdfpy import URDF
import jax.numpy as jnp

import third_party.pyroki.examples.pyroki_snippets as pks


class get_pyroki_model:
    def __init__(self, robot_cfg):
        """
        robot_cfg: an instance of BaseRobotCfg or similar, must contain:
            - urdf_path: str
            - ee_body_name: str (end effector link name)
        """
        self.urdf_path = robot_cfg.urdf_path
        self.ee_link_name = getattr(robot_cfg, "ee_body_name", None)
        if self.ee_link_name is None:
            raise ValueError("robot_cfg must have 'ee_body_name' defined")

        self.urdf = URDF.load(self.urdf_path)
        self.pk_robot = pk.Robot.from_urdf(self.urdf)

    def solve_ik(self, pos_target: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
        """
        Solve IK for a single target.
        """
        # pos = pos_target.detach().cpu().numpy().reshape(3)
        # quat = quat_target.detach().cpu().numpy().reshape(4)

        # Convert PyTorch tensors to JAX arrays (on GPU)
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

    # def solve_trajopt(self, pos_target: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
    #     """
    #     Solve trajectory optimization (fallbacks to IK in this snippet).
    #     """
    #     pos = pos_target.detach().cpu().numpy().reshape(3)
    #     quat = quat_target.detach().cpu().numpy().reshape(4)

    #     solution = pks.solve_ik(  # could be replaced with a real trajopt call
    #         self.pk_robot,
    #         self.ee_link_name,
    #         target_wxyz=quat,
    #         target_position=pos,
    #     )

    #     q_list = np.concatenate([solution, [0.04, 0.04]])
    #     q_tensor = torch.tensor(q_list, dtype=torch.float32)
    #     return q_tensor.cuda() if torch.cuda.is_available() else q_tensor
