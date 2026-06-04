# TODO(jigu): Move to sapien_utils.py
from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
import sapien.core as sapien
from gymnasium import spaces
from ruckig import InputParameter, Ruckig, Trajectory


def parameterize_path(start_qpos, target_qpos, init_v, vlim, alim, jerklim):
    dof = len(start_qpos)
    inp = InputParameter(dof)
    inp.current_position = start_qpos
    if isinstance(init_v, float):
        inp.current_velocity = np.full(dof, init_v)
    else:
        inp.current_velocity = init_v
    inp.current_acceleration = np.zeros(dof)
    inp.target_position = target_qpos
    inp.target_velocity = np.zeros(dof)
    inp.target_acceleration = np.zeros(dof)

    if isinstance(vlim, float):
        vlim = np.full(dof, vlim)
    if isinstance(alim, float):
        alim = np.full(dof, alim)
    if isinstance(jerklim, float):
        jerklim = np.full(dof, jerklim)

    inp.max_velocity = vlim
    inp.max_acceleration = alim
    inp.max_jerk = jerklim
    inp.min_velocity = -vlim
    inp.min_acceleration = -alim

    otg = Ruckig(dof)
    trajectory = Trajectory(dof)
    result = otg.calculate(inp, trajectory)

    return trajectory


def get_joint_indices(articulation: sapien.Articulation, joint_names: Sequence[str]):
    all_joint_names = [x.name for x in articulation.get_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return joint_indices


def get_active_joint_indices(articulation: sapien.Articulation, joint_names: Sequence[str]):
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return joint_indices


def get_joints(articulation: sapien.Articulation, joint_names: Sequence[str]):
    joints = articulation.get_joints()
    joint_indices = get_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]


def get_active_joints(articulation: sapien.Articulation, joint_names: Sequence[str]):
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]


def flatten_action_spaces(action_spaces: Dict[str, spaces.Space]):
    """Flat multiple Box action spaces into a single Box space."""
    action_dims = []
    low = []
    high = []
    action_mapping = OrderedDict()
    offset = 0

    for action_name, action_space in action_spaces.items():
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, (action_name, action_space)
        else:
            raise TypeError(action_space)

        action_dim = action_space.shape[0]
        action_dims.append(action_dim)
        low.append(action_space.low)
        high.append(action_space.high)
        action_mapping[action_name] = (offset, offset + action_dim)
        offset += action_dim

    flat_action_space = spaces.Box(
        low=np.hstack(low),
        high=np.hstack(high),
        shape=[sum(action_dims)],
        dtype=np.float32,
    )

    return flat_action_space, action_mapping
