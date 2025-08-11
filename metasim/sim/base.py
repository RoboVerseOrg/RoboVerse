from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from scenario_cfg.scenario import ScenarioCfg

from metasim.queries.base import BaseQueryType
from metasim.types import Action, DictEnvState, TensorState


class BaseSimHandler(ABC):
    """Base class for simulation handler."""

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        self.scenario = scenario
        self.optional_queries = optional_queries

        ## For quick reference
        self.robots = scenario.robots
        self.cameras = scenario.cameras
        self.objects = scenario.objects
        self._num_envs = scenario.num_envs
        self.decimation = scenario.decimation
        self.headless = scenario.headless
        self.object_dict = {obj.name: obj for obj in self.objects + self.robots}

    def launch(self) -> None:
        """Launch the simulation."""
        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)
        # raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Close the simulation."""
        raise NotImplementedError

    ############################################################
    ## Set states
    ############################################################
    @abstractmethod
    def _set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    def set_states(self, states: TensorState | DictEnvState, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment."""
        self._set_states(states, env_ids)

    # @abstractmethod
    def _set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        """Set the dof targets of the environment.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (list[Action]): The target actions for the robot
        """
        self._set_dof_targets(obj_name, actions)

    ############################################################
    ## Get states
    ############################################################
    @abstractmethod
    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        """Get the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
        raise NotImplementedError

    def get_states(
        self, env_ids: list[int] | None = None, mode: Literal["tensor", "dict"] = "tensor"
    ) -> TensorState | DictEnvState:
        """Get the states of the environment."""
        # TODO: do type change here
        return self._get_states(env_ids)

    ############################################################
    ## Get extra queries
    ############################################################
    def get_extra(self):
        """Get the extra information of the environment."""
        ret_dict = {}
        for query_name, query_type in self.optional_queries.items():
            ret_dict[query_name] = query_type()
        return ret_dict

    ############################################################
    ## Simulate
    ############################################################
    @abstractmethod
    def _simulate(self):
        """Simulate the environment for one time step.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    def simulate(self):
        """Simulate the environment."""
        self._simulate()

    ############################################################
    ## Misc
    ############################################################
    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object.
        For a new simulator, you should implement this method.

        Note:
            Different simulators may have different joint order, but joint names should be the same.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the joint names. Default is True. If True, the joint names are returned in alphabetical order. If False, the joint names are returned in the order defined by the simulator.

        Returns:
            list[str]: A list of joint names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_joint_reindex(self, obj_name: str, inverse: bool = False) -> list[int]:
        """Get the reindexing order for joint indices of a given object. The returned indices can be used to reorder the joints such that they are sorted alphabetically by their names.

        Args:
            obj_name (str): The name of the object.
            inverse (bool): Whether to return the inverse reindexing order. Default is False.

        Returns:
            list[int]: A list of joint indices that specifies the order to sort the joints alphabetically by their names.
               The length of the list matches the number of joints. If ``inverse`` is True, the returned list is inversed, which means they can be used to restore the original order.

        Example:
            Suppose ``obj_name = "h1"``, and the ``h1`` has joints:

            index 0: ``"hip"``

            index 1: ``"knee"``

            index 2: ``"ankle"``

            This function will return: ``[2, 0, 1]``, which corresponds to the alphabetical order:
                ``"ankle"``, ``"hip"``, ``"knee"``.
        """
        if not hasattr(self, "_joint_reindex_cache"):
            self._joint_reindex_cache = {}
            self._joint_reindex_cache_inverse = {}

        if obj_name not in self._joint_reindex_cache:
            origin_joint_names = self._get_joint_names(obj_name, sort=False)
            sorted_joint_names = self._get_joint_names(obj_name, sort=True)
            self._joint_reindex_cache[obj_name] = [origin_joint_names.index(jn) for jn in sorted_joint_names]
            self._joint_reindex_cache_inverse[obj_name] = [sorted_joint_names.index(jn) for jn in origin_joint_names]

        return self._joint_reindex_cache_inverse[obj_name] if inverse else self._joint_reindex_cache[obj_name]

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the body names for a given object.
        For a new simulator, you should implement this method.

        Note:
            Different simulators may have different body order, but body names should be the same.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the body names. Default is True. If True, the body names are returned in alphabetical order. If False, the body names are returned in the order defined by the simulator.

        Returns:
            list[str]: A list of body names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_body_reindex(self, obj_name: str) -> list[int]:
        """Get the reindexing order for body indices of a given object. The returned indices can be used to reorder the bodies such that they are sorted alphabetically by their names.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: A list of body indices that specifies the order to sort the bodies alphabetically by their names.
               The length of the list matches the number of bodies.

        Example:
            Suppose ``obj_name = "h1"``, and the ``h1`` has the following bodies:

                - index 0: ``"torso"``
                - index 1: ``"left_leg"``
                - index 2: ``"right_leg"``

            This function will return: ``[1, 2, 0]``, which corresponds to the alphabetical order:
                ``"left_leg"``, ``"right_leg"``, ``"torso"``.
        """
        if not hasattr(self, "_body_reindex_cache"):
            self._body_reindex_cache = {}

        if obj_name not in self._body_reindex_cache:
            origin_body_names = self._get_body_names(obj_name, sort=False)
            sorted_body_names = self._get_body_names(obj_name, sort=True)
            self._body_reindex_cache[obj_name] = [origin_body_names.index(bn) for bn in sorted_body_names]

        return self._body_reindex_cache[obj_name]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        raise NotImplementedError
