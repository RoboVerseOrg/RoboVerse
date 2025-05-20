from __future__ import annotations

import torch
from loguru import logger as log

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
        self.sensors = scenario.sensors
        self.objects = scenario.objects
        self.checker = scenario.checker
        self.object_dict = {obj.name: obj for obj in self.objects + [self.robot] + self.checker.get_debug_viewers()}
        """A dict mapping object names to object cfg instances. It includes objects, robot, and checker debug viewers."""

    def launch(self) -> None:
        """Launch the simulation."""
        raise NotImplementedError

    ############################################################
    ## Gymnasium main methods
    ############################################################
    def step(self, action: list[Action]) -> tuple[Obs, Reward, Success, TimeOut, Extra]:
        raise NotImplementedError

    def reset(self, env_ids: list[int] | None = None) -> tuple[TensorState, Extra]:
        """Reset the environment.

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
        """Close the simulation."""
        raise NotImplementedError

    ############################################################
    ## Set states
    ############################################################
    def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (list[Action]): The target actions for the robot
        """
        raise NotImplementedError

    def set_pose(self, obj_name: str, pos: torch.Tensor, rot: torch.Tensor, env_ids: list[int] | None = None) -> None:
        states = [{obj_name: {"pos": pos[env_id], "rot": rot[env_id]}} for env_id in range(self.num_envs)]
        self.set_states(states, env_ids=env_ids)

    ############################################################
    ## Get states
    ############################################################
    def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        """Get the states of the environment.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
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
        if obj_name in states.objects:
            return states.objects[obj_name].root_state[:, 7:10]
        elif obj_name in states.robots:
            return states.robots[obj_name].root_state[:, 7:10]
        else:
            raise ValueError(f"Object {obj_name} not found in states")

    def get_pos(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_pos method, which could be slow, please contact the maintainer to"
                " support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        if obj_name in states.objects:
            return states.objects[obj_name].root_state[:, :3]
        elif obj_name in states.robots:
            return states.robots[obj_name].root_state[:, :3]
        else:
            raise ValueError(f"Object {obj_name} not found in states")

    def get_rot(self, obj_name: str, env_ids: list[int] | None = None) -> torch.FloatTensor:
        if self.num_envs > 1:
            log.warning(
                "You are using the unoptimized get_rot method, which could be slow, please contact the maintainer to"
                " support the optimized version if necessary"
            )
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        states = self.get_states(env_ids=env_ids)
        if obj_name in states.objects:
            return states.objects[obj_name].root_state[:, 3:7]
        elif obj_name in states.robots:
            return states.robots[obj_name].root_state[:, 3:7]
        else:
            raise ValueError(f"Object {obj_name} not found in states")

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
        """Step the simulation."""
        raise NotImplementedError

    ############################################################
    ## Utils
    ############################################################
    def refresh_render(self) -> None:
        """Refresh the render."""
        raise NotImplementedError

    ############################################################
    ## Misc
    ############################################################
    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object.

        Note:
            Different simulators may have different joint order, but joint names should be the same.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the joint names. Default is True. If True, the joint names are returned in alphabetical order. If False, the joint names are returned in the order defined by the simulator.

        Returns:
            list[str]: A list of joint names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_joint_reindex(self, obj_name: str) -> list[int]:
        """Get the reindexing order for joint indices of a given object. The returned indices can be used to reorder the joints such that they are sorted alphabetically by their names.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: A list of joint indices that specifies the order to sort the joints alphabetically by their names.
               The length of the list matches the number of joints.

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

        if obj_name not in self._joint_reindex_cache:
            origin_joint_names = self.get_joint_names(obj_name, sort=False)
            sorted_joint_names = self.get_joint_names(obj_name, sort=True)
            self._joint_reindex_cache[obj_name] = [origin_joint_names.index(jn) for jn in sorted_joint_names]

        return self._joint_reindex_cache[obj_name]

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the body names for a given object.

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
            origin_body_names = self.get_body_names(obj_name, sort=False)
            sorted_body_names = self.get_body_names(obj_name, sort=True)
            self._body_reindex_cache[obj_name] = [origin_body_names.index(bn) for bn in sorted_body_names]

        return self._body_reindex_cache[obj_name]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def episode_length_buf(self) -> list[int]:
        """The timestep of each environment, restart from 0 when reset, plus 1 at each step."""
        raise NotImplementedError

    @property
    def actions_cache(self) -> list[Action]:
        """Cache of actions."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError
