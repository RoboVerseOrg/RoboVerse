from __future__ import annotations

import torch

from metasim.sim import BaseSimHandler
from metasim.types import Action, TensorState
from metasim.utils.state import _dof_tensor_to_dict


def adapt_actions(handler: BaseSimHandler, actions: list[Action] | TensorState):
    if isinstance(actions, torch.Tensor):
        if len(actions.shape) == 2:
            actions = actions[0]
        actions = {handler.robot.name: _dof_tensor_to_dict(actions, handler.get_joint_names(handler.robot.name))}
    if isinstance(actions, list):
        actions = actions[0]
    return actions
