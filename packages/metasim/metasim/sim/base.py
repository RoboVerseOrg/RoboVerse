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
        self._actions_cache: CompatActionInput | None = None
        # When True, ``flush_visual_updates`` is deferred — the domain
        # randomization pipeline sets this to batch many visual edits and
        # flush once. Declared on the base so randomizers can set it
        # uniformly regardless of backend.
        self._defer_all_visual_flushes: bool = False

    @property
    def actions_cache(self) -> CompatActionInput | None:
        """Return the most recently submitted action.

        Set by ``set_dof_targets`` and returned here unchanged. Concrete
        backends used to implement this themselves (with subtly different
        attribute names / set times), and ``ParallelHandler`` /
        ``HybridSimHandler`` didn't implement it at all — so
        ``handler.actions_cache`` raised ``AttributeError`` on the
        parallel path even though tests assert it. Centralising the
        cache on the base ensures every handler honours the contract.
        """
        return getattr(self, "_actions_cache", None)

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

    ############################################################
    ## Reproducibility
    ############################################################
    def set_seed(self, seed: int) -> None:
        """Seed the handler's RNG for reproducibility.

        The default implementation seeds Python ``random``, NumPy, and
        Torch's CPU + (if available) CUDA generators. That's enough for
        deterministic ``np.random``-based perturbations and any DR
        callback that uses ``torch.rand``. Backends with extra internal
        RNG (Newton's warp kernels, Sapien physics noise, etc.) should
        override and call ``super().set_seed(seed)`` first.

        Closes the ``env.reset(seed=N)`` reproducibility contract that
        was previously broken: gym would seed its own RNG but ``task.reset``
        never forwarded the seed to the handler, so rollouts were not
        actually deterministic on the simulator side.
        """
        import random as _random

        import numpy as _np

        _random.seed(seed)
        _np.random.seed(seed)
        try:
            import torch as _torch

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except ImportError:  # pragma: no cover — torch is a hard dep
            pass

    # Keys that ``_set_states`` actually writes to the simulator.
    _SET_STATES_VALID_OBJECT_KEYS = frozenset({"pos", "rot", "dof_pos"})
    _SET_STATES_VALID_ROBOT_KEYS = frozenset({"pos", "rot", "dof_pos"})
    # Velocity fields are populated by ``get_states`` but no backend honours
    # them in ``_set_states`` today (MuJoCo zeros qvel, Sapien3's
    # ``set_velocity`` calls are commented out). Listing them here makes the
    # silent drop visible instead of letting callers believe they are
    # initialising momentum.
    _SET_STATES_READ_ONLY_KEYS = frozenset({"vel", "ang_vel", "dof_vel"})
    # Control inputs that belong in ``set_dof_targets``, not ``set_states``.
    _SET_STATES_CONTROL_KEYS = frozenset({"dof_pos_target", "dof_vel_target", "dof_torque"})

    def _warn_set_states_keys(self, states: DictStateBatch) -> None:
        """Surface dict keys that ``_set_states`` will silently ignore.

        Without this check a caller can pass e.g. ``dof_pos_target`` expecting
        the joints to move and get a silent no-op — a class of bug that has
        already cost downstream users multiple failed experiments. We warn
        once per (key, role) per handler instance.

        Also flags partial pose input (only ``pos`` or only ``rot``):
        MuJoCo silently substitutes a default for the missing component
        (overwriting whatever value the entity had), while Sapien3 raises
        ``KeyError`` from ``Pose(p=val["pos"], q=val["rot"])``. Same input,
        divergent behaviour. Warn so the caller can pass both.
        """
        if not isinstance(states, list):
            return
        seen: set[tuple[str, str]] = getattr(self, "_set_states_warned_keys", set())
        partial_pose_seen: set[tuple[str, str, str]] = getattr(self, "_set_states_partial_pose_warned", set())
        for env_state in states:
            if not isinstance(env_state, dict):
                continue
            for role, valid in (
                ("objects", self._SET_STATES_VALID_OBJECT_KEYS),
                ("robots", self._SET_STATES_VALID_ROBOT_KEYS),
            ):
                bucket = env_state.get(role)
                if not isinstance(bucket, dict):
                    continue
                for entity_name, entity_state in bucket.items():
                    if not isinstance(entity_state, dict):
                        continue
                    # Partial-pose check — pos XOR rot.
                    has_pos = "pos" in entity_state
                    has_rot = "rot" in entity_state
                    if has_pos != has_rot:
                        missing = "rot" if has_pos else "pos"
                        pp_key = (role, entity_name, missing)
                        if pp_key not in partial_pose_seen:
                            partial_pose_seen.add(pp_key)
                            log.warning(
                                f"set_states for {role}:{entity_name} has '{'pos' if has_pos else 'rot'}' "
                                f"without '{missing}' — cross-backend behaviour diverges: MuJoCo silently "
                                f"substitutes a default for the missing component, Sapien3 raises KeyError. "
                                f"Pass both pos and rot together, or omit both."
                            )
                    for key in entity_state.keys():
                        if key in valid:
                            continue
                        cache_key = (role, key)
                        if cache_key in seen:
                            continue
                        seen.add(cache_key)
                        if key in self._SET_STATES_CONTROL_KEYS:
                            log.warning(
                                f"set_states received '{key}' under '{role}' — this is a control "
                                f"input, not a state. set_states will NOT apply it. "
                                f"Use set_dof_targets(...) to drive joints to a target."
                            )
                        elif key in self._SET_STATES_READ_ONLY_KEYS:
                            log.warning(
                                f"set_states received '{key}' under '{role}' — velocity fields are "
                                f"populated by get_states for inspection, but no backend currently "
                                f"writes them in _set_states (MuJoCo zeros qvel, Sapien3 ignores). "
                                f"The value will be dropped. Pass it through simulate() with a "
                                f"non-zero initial action instead, or open a feature request for "
                                f"backend-side velocity initialisation."
                            )
                        else:
                            log.warning(
                                f"set_states received unknown key '{key}' under '{role}' — "
                                f"it will be ignored. Valid keys: {sorted(valid)}."
                            )
        self._set_states_warned_keys = seen
        self._set_states_partial_pose_warned = partial_pose_seen

    def set_states(self, states: TensorState | DictStateBatch, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.

        Input is first normalised to the shape the backend declared via
        ``_set_states_input_type``. Cache invalidation runs *after*
        ``_set_states`` returns (via try/finally): if anything inside the
        mutation re-enters ``get_states``, the pre-mutation cache must
        not survive past it.

        Also runs ``_warn_set_states_keys`` against the raw dict input so
        silent-drop / partial-pose / read-only-key mistakes surface as a
        one-shot warning (see :py:meth:`_warn_set_states_keys`).
        """
        self._warn_set_states_keys(states)
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

    # Keys ``_set_dof_targets`` honours under each robot dict.
    _SET_DOF_TARGETS_VALID_KEYS = frozenset({"dof_pos_target", "dof_vel_target", "dof_torque", "dof_effort_target"})

    def _warn_set_dof_targets(self, actions: CompatActionInput) -> None:
        """Surface unknown robot names or joint names in dict actions.

        ``set_dof_targets`` with a list-of-dicts action looks like::

            [{"robot_a": {"dof_pos_target": {"joint1": 0.5, ...}}}]

        Every backend's handler walks the per-robot ``get_joint_names``
        list and silently skips any joint not in it. A typo like
        ``panda_joint99`` therefore disappears with no feedback.
        Tensor-input actions don't have this problem — they're indexed,
        not name-based — so the bug only bites the dict-API path.

        Warn once per ``(robot, joint)`` or ``(unknown_robot)`` so
        callers see the silent drop. Robot-name validation also surfaces
        callers that pass actions for a robot that the scenario does
        not include.
        """
        if not isinstance(actions, list):
            return
        robots = getattr(self, "robots", None) or []
        known_robots = {robot.name for robot in robots}
        # Build a per-robot known-joint set lazily so the warning is
        # cheap on the hot path when no unknown name is seen.
        joint_names_cache: dict[str, set[str]] = {}
        seen_robots: set[str] = getattr(self, "_set_dof_targets_warned_robots", set())
        seen_joints: set[tuple[str, str]] = getattr(self, "_set_dof_targets_warned_joints", set())
        seen_keys: set[tuple[str, str]] = getattr(self, "_set_dof_targets_warned_keys", set())
        for env_action in actions:
            if not isinstance(env_action, dict):
                continue
            for robot_name, robot_action in env_action.items():
                if robot_name not in known_robots:
                    if robot_name not in seen_robots:
                        seen_robots.add(robot_name)
                        log.warning(
                            f"set_dof_targets received action for unknown robot "
                            f"'{robot_name}'. Known robots: {sorted(known_robots)}. "
                            f"The action will be silently dropped by every backend."
                        )
                    continue
                if not isinstance(robot_action, dict):
                    continue
                # Unknown top-level keys (anything not in dof_*_target / dof_torque).
                for key in robot_action:
                    if key in self._SET_DOF_TARGETS_VALID_KEYS:
                        continue
                    cache = (robot_name, key)
                    if cache in seen_keys:
                        continue
                    seen_keys.add(cache)
                    log.warning(
                        f"set_dof_targets for robot '{robot_name}' got unknown key "
                        f"'{key}' — it will be ignored. Valid keys: "
                        f"{sorted(self._SET_DOF_TARGETS_VALID_KEYS)}."
                    )
                joint_targets = robot_action.get("dof_pos_target") or {}
                if not joint_targets:
                    continue
                if robot_name not in joint_names_cache:
                    try:
                        joint_names_cache[robot_name] = set(self.get_joint_names(robot_name, sort=False))
                    except Exception:
                        # Handler may not be fully launched yet; skip name validation.
                        joint_names_cache[robot_name] = set()
                known_joints = joint_names_cache[robot_name]
                if not known_joints:
                    # Handler can't resolve joint names yet (pre-launch); skip.
                    continue
                for joint_name in joint_targets:
                    if joint_name in known_joints:
                        continue
                    cache = (robot_name, joint_name)
                    if cache in seen_joints:
                        continue
                    seen_joints.add(cache)
                    log.warning(
                        f"set_dof_targets for robot '{robot_name}' got unknown joint "
                        f"'{joint_name}' — it will be silently dropped. Known joints: "
                        f"{sorted(known_joints)}."
                    )
        self._set_dof_targets_warned_robots = seen_robots
        self._set_dof_targets_warned_joints = seen_joints
        self._set_dof_targets_warned_keys = seen_keys

    def set_dof_targets(self, actions: CompatActionInput) -> None:
        """Set the dof targets of the robot.

        Dict actions are name-based. Tensor actions use ``get_joint_names(robot.name, sort=True)``
        within each robot slice. Backends may remap from handler/API order to simulator-local
        order internally.

        Args:
            actions: The target actions for the robot.
        """
        self._warn_set_dof_targets(actions)
        # Centralised ``actions_cache`` contract — see the @property above.
        # Concrete backends keep their own ``self._actions_cache = actions``
        # writes for back-compat with their internal use, but the public
        # contract is now owned by the base.
        self._actions_cache = actions
        # Backends like MuJoCo write actuator ctrl here, and ``get_states`` reads ctrl
        # back as ``joint_pos_target``. Without invalidation, the cached state held a
        # pre-action joint_pos_target until the next simulate() — silent staleness.
        self._invalidate_state_caches()
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

        Subset requests (``env_ids is not None``) bypass the cache entirely. The
        shared cache holds the full env set; backends that honour ``env_ids``
        (``ParallelHandler`` and mjx/genesis/newton) return a real subset, so
        caching it would make a later full ``get_states()`` return the stale
        subset (and vice versa). Subset callers already hit a fresh fetch today
        (``reset`` invalidates first), so not caching them costs nothing.
        """
        if env_ids is not None:
            result = self._get_states(env_ids=env_ids)
            if result is None:
                return None
            if mode == "tensor":
                return result if isinstance(result, TensorState) else list_state_to_tensor(self, result)
            return state_tensor_to_nested(self, result) if isinstance(result, TensorState) else result

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
