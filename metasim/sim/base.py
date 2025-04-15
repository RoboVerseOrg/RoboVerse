from __future__ import annotations

import torch
from loguru import logger as log

from metasim.cfg.objects import BaseObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success, TimeOut
from metasim.utils.state import TensorState, state_tensor_to_nested


class BaseSimHandler:
    """Base class for simulation handler."""

    def __init__(self, scenario: ScenarioCfg):
        ## Overwrite scenario with task, TODO: this should happen in scenario class post_init
        if scenario.task is not None:
            scenario.objects = scenario.task.objects
            scenario.checker = scenario.task.checker
            scenario.decimation = scenario.task.decimation
            scenario.episode_length = scenario.task.episode_length

        self.scenario = scenario
        self._num_envs = scenario.num_envs
        self.headless = scenario.headless

        ## For quick reference
        self.task = scenario.task
        self.robot = scenario.robot
        self.cameras = scenario.cameras
        self.objects = scenario.objects
        self.checker = scenario.checker
        self.object_dict = {obj.name: obj for obj in self.objects + [self.robot] + self.checker.get_debug_viewers()}
        """A dict mapping object names to object cfg instances. It includes objects, robot, and checker debug viewers."""

    def launch(self) -> None:
        raise NotImplementedError

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        raise NotImplementedError

    def reset(self, env_ids: list[int] | None = None) -> tuple[TensorState, Extra]:
        """
        Reset the environment.

        Args:
            env_ids: The indices of the environments to reset. If None, all environments are reset.

        Return:
            obs: The observation of the environment. Currently all the environments are returned. Do we need to return only the reset environments?
            extra: Extra information. Currently is empty.
        """
        raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    ############################################################
    ## Set states
    ############################################################
    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        raise NotImplementedError

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        raise NotImplementedError

    def set_pose(self, obj_name: str, pos: torch.Tensor, rot: torch.Tensor, env_ids: list[int] | None = None) -> None:
        states = [{obj_name: {"pos": pos[env_id], "rot": rot[env_id]}} for env_id in range(self.num_envs)]
        self.set_states(states, env_ids=env_ids)

    ############################################################
    ## Get states
    ############################################################
    def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        raise NotImplementedError

    def get_vel(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_pos method, which could be slow, please contact the maintainer to"
                " support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        states = state_tensor_to_nested(self, states)
        return torch.stack([{**env_state["objects"], **env_state["robots"]}[obj_name]["vel"] for env_state in states])

    def get_pos(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_pos method, which could be slow, please contact the maintainer to"
                " support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        states = state_tensor_to_nested(self, states)
        return torch.stack([{**env_state["objects"], **env_state["robots"]}[obj_name]["pos"] for env_state in states])

    def get_rot(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_rot method, which could be slow, please contact the maintainer to"
                " support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        states = state_tensor_to_nested(self, states)
        return torch.stack([{**env_state["objects"], **env_state["robots"]}[obj_name]["rot"] for env_state in states])

    def get_dof_pos(self, obj_name: str, joint_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_dof_pos method, which could be slow, please contact the maintainer"
                " to support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        states = state_tensor_to_nested(self, states)
        return torch.tensor([
            {**env_state["objects"], **env_state["robots"]}[obj_name]["dof_pos"][joint_name] for env_state in states
        ])

    ############################################################
    ## Simulate
    ############################################################
    def simulate(self):
        raise NotImplementedError

    ############################################################
    ## Utils
    ############################################################
    def refresh_render(self) -> None:
        raise NotImplementedError

    ############################################################
    ## Misc
    ############################################################
    def get_object_joint_names(self, object: BaseObjCfg) -> list[str]:
        """Get the joint names for a specified object in the order of the simulator default joint order. For same object, different simulator may have different joint order, but joint names are the same.

        Args:
            object (BaseObjCfg): The target object.

        Returns:
            list[str]: A list of strings including the joint names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_joint_reindex(self, obj_name: str) -> list[int]:
        """Get the reindex of the joint names for a specified object. After reindexing, the joint order is alphabetical. For same object, different simulator may have different joint order, thus have different reindex.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: A list of integers including the reindex of the joint names.
        """
        if not hasattr(self, "_joint_reindex_cache"):
            self._joint_reindex_cache = {}

        if obj_name not in self._joint_reindex_cache:
            obj_cfg = self.object_dict[obj_name]
            origin_joint_names = self.get_object_joint_names(obj_cfg)
            sorted_joint_names = sorted(origin_joint_names)
            self._joint_reindex_cache[obj_name] = [origin_joint_names.index(jn) for jn in sorted_joint_names]

        return self._joint_reindex_cache[obj_name]

    def get_body_names(self, obj_name: str) -> list[str]:
        """Get the body names for a specified object in the order of the simulator default body order. For same object, different simulator may have different body order, but body names are the same.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[str]: A list of strings including the body names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_body_reindex(self, obj_name: str) -> list[int]:
        """Get the reindex of the body names for a specified object. After reindexing, the body order is alphabetical. For same object, different simulator may have different body order, thus have different reindex.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: A list of integers including the reindex of the body names.
        """
        if not hasattr(self, "_body_reindex_cache"):
            self._body_reindex_cache = {}

        if obj_name not in self._body_reindex_cache:
            origin_body_names = self.get_body_names(obj_name)
            sorted_body_names = sorted(origin_body_names)
            self._body_reindex_cache[obj_name] = [origin_body_names.index(bn) for bn in sorted_body_names]

        return self._body_reindex_cache[obj_name]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def episode_length_buf(self) -> list[int]:
        """
        The timestep of each environment, restart from 0 when reset, plus 1 at each step.
        """
        raise NotImplementedError

    @property
    def actions_cache(self) -> list[Action]:
        """
        Cache of actions.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError
