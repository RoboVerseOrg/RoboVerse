from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

import torch

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

from loguru import logger as log

from metasim.queries.base import BaseQueryType
from metasim.types import CompatActionInput, DictStateBatch, StateMode, StateOutput, TensorState
from metasim.utils.gs_util import quaternion_multiply
from metasim.utils.state import list_state_to_tensor, state_tensor_to_nested

try:
    from robo_splatter.models.basic import GSInstance, RenderConfig
    from robo_splatter.models.gaussians import RigidsGaussians
    from robo_splatter.render.scenes import Scene

    ROBO_SPLATTER_AVAILABLE = True
except ImportError:
    ROBO_SPLATTER_AVAILABLE = False


class BaseSimHandler(ABC):
    """Base class for simulation handler.

    Contract summary (enforced by ``@abstractmethod`` unless noted):

    Required by every backend:
        - ``_set_states(states, env_ids=None)``
        - ``_set_dof_targets(actions)``
        - ``_get_states(env_ids=None)``
        - ``_simulate()``

    Required by every backend, enforced only with a runtime
    ``NotImplementedError`` (kept un-decorated to avoid breaking
    historical handler stubs that subclassed without the method):
        - ``close()``
        - ``device`` (property)

    Optional / sensible defaults provided:
        - ``render()`` — backends without a viewer raise; that's expected.
        - ``refresh_render()`` — no-op default; hybrid backends override.
        - ``_get_joint_names(obj_name, sort=True)`` — returns ``[]``
          (consistent with the docstring saying "for non-articulation
          objects, return an empty list").
        - ``_get_body_names(obj_name, sort=True)`` — returns ``[]``.
        - ``flush_visual_updates(**kwargs)`` — no-op; backends with
          independent renderers override.

    Backends also declare what their ``_set_states`` accepts via the
    class attribute ``_set_states_input_type``. ``"both"`` (the default)
    passes the caller's input through untouched; ``"dict"`` means the
    handler only consumes ``list[DictEnvState]`` and the base converts
    a ``TensorState`` for it.
    """

    _set_states_input_type: Literal["dict", "both"] = "both"

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        self.scenario = scenario
        self.optional_queries = optional_queries
        scenario.check_assets()  # check if all assets are available

        ## For quick reference
        self.robots = scenario.robots
        self.cameras = scenario.cameras
        self.objects = scenario.objects
        self.lights = scenario.lights if hasattr(scenario, "lights") else []
        self.gs_background = None

        self._num_envs = scenario.num_envs
        self.decimation = scenario.decimation
        self.headless = scenario.headless
        self.object_dict = {obj.name: obj for obj in self.objects + self.robots}
        self._tensor_state_cache: TensorState | None = None
        self._dict_state_cache: DictStateBatch | None = None

        # When True, ``flush_visual_updates`` is deferred — the domain
        # randomization pipeline sets this to batch many visual edits and
        # flush once. Declared on the base so randomizers can set it
        # uniformly regardless of backend.
        self._defer_all_visual_flushes: bool = False

    def flush_visual_updates(self, *, wait_for_materials: bool = False, settle_passes: int = 2) -> None:
        """Settle pending visual edits (no-op by default).

        Backends that drive a renderer independently from the physics
        step (Isaac Sim, hybrid Blender) override this to step the render
        pipeline forward; physics-only backends keep the no-op.
        """
        return

    def launch(self) -> None:
        """Launch the simulation."""
        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)
        # raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def refresh_render(self) -> None:
        """Force the render pipeline to update from the current simulation state.

        Default is a no-op. Handlers that drive a renderer independently from the
        physics step (e.g. hybrid render backends) should override this.
        """
        return

    def close(self) -> None:
        """Close the simulation."""
        raise NotImplementedError

    ############################################################
    ## Set states
    ############################################################
    @abstractmethod
    def _set_states(self, states: TensorState | DictStateBatch, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    def _invalidate_state_caches(self) -> None:
        """Mark both tensor and dict state caches as stale.

        Called automatically by ``set_states``/``simulate``. Subclasses or
        custom mutation paths (e.g. randomizers that poke the underlying
        sim outside the public API) must call ``invalidate_state_caches``
        themselves, since the base handler cannot detect those edits.
        """
        self._tensor_state_cache = None
        self._dict_state_cache = None

    def invalidate_state_caches(self) -> None:
        """Public alias for ``_invalidate_state_caches``.

        Use this from any code path that mutates physics state outside
        ``set_states``/``simulate`` so subsequent ``get_states`` calls
        refetch from the simulator instead of returning stale values.
        """
        self._invalidate_state_caches()

    def set_states(self, states: TensorState | DictStateBatch, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.

        Input is first normalised to the shape the backend declared via
        ``_set_states_input_type``. Cache invalidation runs *after*
        ``_set_states`` returns (via try/finally): if anything inside the
        mutation re-enters ``get_states``, the pre-mutation cache must
        not survive past it.
        """
        normalised = self._normalise_set_states_input(states)
        try:
            self._set_states(normalised, env_ids)
        finally:
            self._invalidate_state_caches()

    def _normalise_set_states_input(self, states):
        """Coerce ``states`` to the shape declared by ``_set_states_input_type``."""
        wanted = type(self)._set_states_input_type
        if wanted == "dict" and isinstance(states, TensorState):
            return state_tensor_to_nested(self, states)
        return states

    @abstractmethod
    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        """Set the dof targets of the environment.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    def set_dof_targets(self, actions: CompatActionInput) -> None:
        """Set the dof targets of the robot.

        Dict actions are name-based. Tensor actions use ``get_joint_names(robot.name, sort=True)``
        within each robot slice. Backends may remap from handler/API order to simulator-local
        order internally.

        Args:
            actions: The target actions for the robot.
        """
        self._set_dof_targets(actions)

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
            TensorState: The tensorized state of the environment.
        """
        raise NotImplementedError

    @overload
    def get_states(self, env_ids: list[int] | None = None, mode: Literal["tensor"] = "tensor") -> TensorState: ...

    @overload
    def get_states(self, env_ids: list[int] | None = None, mode: Literal["dict"] = "dict") -> DictStateBatch: ...

    def get_states(self, env_ids: list[int] | None = None, mode: StateMode = "dict") -> StateOutput:
        """Get the states of the environment.

        Maintains independent tensor and dict caches so alternating modes does not
        destroy either representation. A cache miss on the requested mode is filled
        lazily by converting from the other cache.
        """
        # Fetch fresh from sim only if both caches are stale.
        if self._tensor_state_cache is None and self._dict_state_cache is None:
            result = self._get_states(env_ids=env_ids)
            if result is None:
                return None  # handler does not implement _get_states (e.g. stub handlers)
            if isinstance(result, TensorState):
                self._tensor_state_cache = result
            else:
                self._dict_state_cache = result

        if mode == "tensor":
            if self._tensor_state_cache is None:
                self._tensor_state_cache = list_state_to_tensor(self, self._dict_state_cache)
            return self._tensor_state_cache
        # mode == "dict"
        if self._dict_state_cache is None:
            self._dict_state_cache = state_tensor_to_nested(self, self._tensor_state_cache)
        return self._dict_state_cache

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
        """Simulate the environment.

        See ``set_states`` for the rationale on invalidating after the
        mutation rather than before.
        """
        try:
            self._simulate()
        finally:
            self._invalidate_state_caches()

    ############################################################
    ## Misc
    ############################################################
    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object.

        Default implementation returns an empty list, matching the
        documented contract "for non-articulation objects, return an
        empty list". Articulation-aware backends override.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the joint names alphabetically.

        Returns:
            list[str]: A list of joint names, or ``[]`` for non-articulations.
        """
        return []

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object."""
        return self._get_joint_names(obj_name, sort)

    def get_action_joint_names(self) -> dict[str, list[str]]:
        """Get the handler/API joint order used for tensor actions."""
        return {robot.name: self.get_joint_names(robot.name, sort=True) for robot in self.robots}

    def get_action_dim(self) -> int:
        """Get the total flattened action dimension across all robots."""
        return sum(len(joint_names) for joint_names in self.get_action_joint_names().values())

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

        Default implementation returns an empty list, matching the
        documented contract "for non-articulation objects, return an
        empty list". Articulation-aware backends override.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the body names alphabetically.

        Returns:
            list[str]: A list of body names, or ``[]`` for non-articulations.
        """
        return []

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the body names for a given object."""
        return self._get_body_names(obj_name, sort)

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

    ############################################################
    ## GS Renderer
    ############################################################
    def _get_camera_params(self, camera):
        """Get the camera parameters for GS rendering.
        For a new simulator, you should implement this method.
        Args:
            camera: PinholeCameraCfg object

        Returns:
            Ks: (3, 3) intrinsic matrix
            c2w: (4, 4) camera-to-world transformation matrix
        """
        raise NotImplementedError

    def _build_gs_background(self):
        """Initialize GS background renderer if enabled in scenario config."""
        if self.scenario.gs_scene is None or not self.scenario.gs_scene.with_gs_background:
            self.gs_background = None
            return

        if not ROBO_SPLATTER_AVAILABLE:
            log.error("GS background enabled but RoboSplatter not available.")
            self.gs_background = None
            return
        # Parse pose transformation
        if self.scenario.gs_scene.gs_background_pose_tum is not None:
            x, y, z, qx, qy, qz, qw = self.scenario.gs_scene.gs_background_pose_tum
        else:
            x, y, z, qx, qy, qz, qw = 0, 0, 0, 0, 0, 0, 1

        # Apply coordinate transform
        qx, qy, qz, qw = quaternion_multiply([qx, qy, qz, qw], [0.7071, 0, 0, 0.7071])
        init_pose = torch.tensor([x, y, z, qx, qy, qz, qw], dtype=torch.float32).cpu()

        # Load GS model
        gs_model = RigidsGaussians(
            instances={0: GSInstance(gs_model_path=self.scenario.gs_scene.gs_background_path, init_pose=init_pose)},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.gs_background = Scene(render_config=RenderConfig(), foreground_models=gs_model)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        raise NotImplementedError
