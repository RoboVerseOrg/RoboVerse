"""Unified visualization wrapper for RLTaskEnv.

This module provides a unified wrapper for RLTaskEnv that supports both Rerun and Viser
visualization simultaneously or individually.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Check for Rerun availability
try:
    import rerun as rr  # noqa: F401

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

# Check for Viser availability
try:
    import viser  # noqa: F401

    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False


class TaskVizWrapper:
    """Unified wrapper for RLTaskEnv with Rerun and/or Viser visualization.

    This wrapper supports both Rerun and Viser visualization simultaneously or individually.
    Only renders the first environment for simplicity. Designed for RL training integration.

    Args:
        task_env: RLTaskEnv or similar environment with handler
        use_rerun: Whether to enable Rerun visualization (default: False)
        use_viser: Whether to enable Viser visualization (default: False)
        rerun_app_name: Application name for Rerun (default: "RoboVerse Training")
        rerun_spawn: Whether to spawn Rerun viewer automatically (default: True)
        viser_port: Port for Viser server (default: 8080)
        update_freq: Update visualization every N steps to reduce resource usage (default: 10)
    """

    def __init__(
        self,
        task_env,
        use_rerun: bool = False,
        use_viser: bool = False,
        rerun_app_name: str = "RoboVerse Training",
        rerun_spawn: bool = True,
        viser_port: int = 8080,
        update_freq: int = 10,
    ):
        if use_rerun and not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")

        if use_viser and not VISER_AVAILABLE:
            raise ImportError("viser is required. Install with: pip install viser")

        if not use_rerun and not use_viser:
            logger.warning("Neither Rerun nor Viser is enabled. Wrapper will have no effect.")

        self.env = task_env
        self.use_rerun = use_rerun
        self.use_viser = use_viser
        self.update_freq = update_freq
        self.step_count = 0
        self.handler = self._get_handler()

        # Visualizers
        self.rerun_visualizer = None
        self.viser_visualizer = None

        # Setup visualizations
        if use_rerun:
            self._setup_rerun(rerun_app_name, rerun_spawn)

        if use_viser:
            self._setup_viser(viser_port)

    def _get_handler(self):
        """Get the actual handler object, supporting both gym and non-gym environments."""
        # Try gym environment path first (gym_vec.task_env.handler)
        if hasattr(self.env, "task_env"):
            if hasattr(self.env.task_env, "handler") and self.env.task_env.handler is not None:
                return self.env.task_env.handler

        # Fall back to direct handler access (env.handler)
        if hasattr(self.env, "handler") and self.env.handler is not None:
            return self.env.handler

        return None

    def _setup_rerun(self, app_name: str, spawn: bool):
        """Setup Rerun visualization."""
        if self.handler is None:
            logger.warning("No handler found, skipping Rerun visualization setup")
            return

        try:
            from metasim.utils.rerun.rerun_util import RerunVisualizer
            from metasim.utils.state import state_tensor_to_nested

            # Download URDF files
            self._download_urdf_files()

            # Initialize visualizer
            self.rerun_visualizer = RerunVisualizer(app_name=app_name, spawn=spawn)

            # Get initial states
            obs = self.handler.get_states(mode="tensor")
            env_states = state_tensor_to_nested(self.handler, obs)

            if env_states and len(env_states) > 0:
                state = env_states[0]
                object_states = state.get("objects", {})
                robot_states = state.get("robots", {})

                # Visualize objects
                for obj in self.handler.objects:
                    obj_state = self._extract_state(object_states.get(obj.name, {}))
                    self.rerun_visualizer.visualize_item(obj, obj.name, obj_state)

                # Visualize robots
                for robot in self.handler.robots:
                    robot_state = self._extract_state(robot_states.get(robot.name, {}))
                    self.rerun_visualizer.visualize_item(robot, robot.name, robot_state)

            logger.info("Rerun visualization initialized")

        except Exception as e:
            logger.error(f"Failed to setup Rerun visualization: {e}")
            import traceback

            traceback.print_exc()
            self.rerun_visualizer = None

    def _setup_viser(self, port: int):
        """Setup Viser visualization."""
        if self.handler is None:
            logger.warning("No handler found, skipping Viser visualization setup")
            return

        try:
            from metasim.utils.viser.viser_util import ViserVisualizer

            self.viser_visualizer = ViserVisualizer(port=port)
            self.viser_visualizer.add_grid()
            self.viser_visualizer.add_frame("/world_frame")

            # Download URDF files for visualization
            self._download_urdf_files()

            # Get initial states using handler
            obs = self.handler.get_states(mode="tensor")

            # Extract initial states for first environment only
            if hasattr(obs, "objects") and hasattr(obs, "robots"):
                default_object_states = self._extract_states_from_obs(obs, "objects")
                default_robot_states = self._extract_states_from_obs(obs, "robots")

                # Visualize all objects and robots
                # Try to use scenario first, fall back to handler
                objects = None
                robots = None
                if hasattr(self.env, "scenario"):
                    objects = getattr(self.env.scenario, "objects", None)
                    robots = getattr(self.env.scenario, "robots", None)

                if objects is None and self.handler.objects:
                    objects = self.handler.objects
                if robots is None and self.handler.robots:
                    robots = self.handler.robots

                if objects:
                    self.viser_visualizer.visualize_scenario_items(objects, default_object_states)

                if robots:
                    self.viser_visualizer.visualize_scenario_items(robots, default_robot_states)

            # Setup camera
            self.viser_visualizer.enable_camera_controls(
                initial_position=[1.5, -1.5, 1.5],
                render_width=1024,
                render_height=1024,
                look_at_position=[0, 0, 0],
                initial_fov=71.28,
            )

            logger.info("Viser visualization initialized")

        except Exception as e:
            logger.error(f"Failed to setup Viser visualization: {e}")
            import traceback

            traceback.print_exc()
            self.viser_visualizer = None

    def _download_urdf_files(self):
        """Download URDF files for all objects and robots."""
        from metasim.utils.hf_util import check_and_download_recursive

        urdf_paths = []

        for obj in self.handler.objects:
            if hasattr(obj, "urdf_path") and obj.urdf_path:
                urdf_paths.append(obj.urdf_path)

        for robot in self.handler.robots:
            if hasattr(robot, "urdf_path") and robot.urdf_path:
                urdf_paths.append(robot.urdf_path)

        if urdf_paths:
            check_and_download_recursive(urdf_paths, n_processes=16)

    def _extract_state(self, state_dict: dict) -> dict:
        """Extract state from nested dict format."""
        result = {}

        if "pos" in state_dict and state_dict["pos"] is not None:
            pos = state_dict["pos"]
            if hasattr(pos, "cpu"):
                pos = pos.cpu().numpy()
            result["pos"] = list(pos)

        if "rot" in state_dict and state_dict["rot"] is not None:
            rot = state_dict["rot"]
            if hasattr(rot, "cpu"):
                rot = rot.cpu().numpy()
            result["rot"] = list(rot)

        if "dof_pos" in state_dict and state_dict["dof_pos"] is not None:
            result["dof_pos"] = state_dict["dof_pos"]

        return result

    def _extract_states_from_obs(self, obs, key):
        """Extract states from observation tensor (first environment only).

        Args:
            obs: TensorState observation
            key: "objects" or "robots"

        Returns:
            dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
        """
        if not hasattr(obs, key):
            return {}

        result = {}
        items = getattr(obs, key)

        for name, item in items.items():
            state_dict = {}

            # Extract position and rotation from root_state (first 7 values of first env)
            if hasattr(item, "root_state") and item.root_state is not None:
                root_state = item.root_state[0]  # First environment only
                state_dict["pos"] = root_state[:3].cpu().numpy().tolist()
                state_dict["rot"] = root_state[3:7].cpu().numpy().tolist()

            # Extract joint positions (first environment only)
            if hasattr(item, "joint_pos") and item.joint_pos is not None:
                if self.handler is not None:
                    joint_names = self.handler._get_joint_names(name, sort=True)
                    state_dict["dof_pos"] = {
                        joint_names[i]: item.joint_pos[0, i].item() for i in range(len(joint_names))
                    }

            result[name] = state_dict

        return result

    def _update_visualization(self):
        """Update visualization with current state."""
        if self.handler is None:
            return

        try:
            from metasim.utils.state import state_tensor_to_nested

            obs = self.handler.get_states(mode="tensor")

            # Update Rerun visualization
            if self.rerun_visualizer is not None:
                env_states = state_tensor_to_nested(self.handler, obs)

                if env_states and len(env_states) > 0:
                    state = env_states[0]
                    object_states = state.get("objects", {})
                    robot_states = state.get("robots", {})

                    # Update objects
                    for obj in self.handler.objects:
                        if obj.name in object_states:
                            obj_state = self._extract_state(object_states[obj.name])
                            self.rerun_visualizer.update_item_pose(obj.name, obj_state)

                    # Update robots
                    for robot in self.handler.robots:
                        if robot.name in robot_states:
                            robot_state = self._extract_state(robot_states[robot.name])
                            self.rerun_visualizer.update_item_pose(robot.name, robot_state)

                    # Update time step
                    self.rerun_visualizer.set_time(self.step_count)

            # Update Viser visualization
            if self.viser_visualizer is not None:
                if hasattr(obs, "objects") and hasattr(obs, "robots"):
                    # Update objects from first environment
                    object_states = self._extract_states_from_obs(obs, "objects")
                    for name, state in object_states.items():
                        self.viser_visualizer.update_item_pose(name, state)

                    # Update robots from first environment
                    robot_states = self._extract_states_from_obs(obs, "robots")
                    for name, state in robot_states.items():
                        self.viser_visualizer.update_item_pose(name, state)

                    self.viser_visualizer.refresh_camera_view()

        except Exception as e:
            logger.debug(f"Visualization update failed: {e}")

    def step(self, action):
        """Step the environment and update visualization."""
        result = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            self._update_visualization()

        return result

    def reset(self, **kwargs):
        """Reset the environment and update visualization."""
        result = self.env.reset(**kwargs)
        self._update_visualization()
        return result

    def render(self, mode="human"):
        """Render the environment."""
        return None

    def close(self):
        """Close environment and visualizers."""
        if self.rerun_visualizer is not None:
            self.rerun_visualizer.close()
        # Note: ViserVisualizer doesn't have a close method
        # The server will be cleaned up when the object is garbage collected
        if hasattr(self.env, "close"):
            self.env.close()

    def __len__(self) -> int:
        """Return number of environments."""
        return self.env.num_envs if hasattr(self.env, "num_envs") else 1

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the wrapped environment."""
        return getattr(self.env, name)
