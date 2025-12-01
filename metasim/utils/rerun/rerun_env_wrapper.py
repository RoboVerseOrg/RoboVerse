"""Rerun environment wrapper for RoboVerse.

This module provides a wrapper for RLTaskEnv that adds real-time Rerun visualization
during training and evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class TaskRerunWrapper:
    """Simple wrapper for RLTaskEnv with real-time Rerun visualization.

    Only renders the first environment for simplicity. Designed for RL training integration.

    Args:
        task_env: RLTaskEnv or similar environment with handler
        app_name: Application name for Rerun (default: "RoboVerse Training")
        update_freq: Update visualization every N steps to reduce resource usage (default: 10)
        spawn: Whether to spawn the Rerun viewer automatically (default: True)
    """

    def __init__(
        self,
        task_env,
        app_name: str = "RoboVerse Training",
        update_freq: int = 10,
        spawn: bool = True,
    ):
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")

        self.env = task_env
        self.update_freq = update_freq
        self.step_count = 0
        self.handler = self._get_handler()
        self.visualizer = None

        self._setup_visualization(app_name, spawn)

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

    def _setup_visualization(self, app_name: str, spawn: bool):
        """Setup Rerun visualization."""
        if self.handler is None:
            print("[TaskRerunWrapper] Warning: No handler found, skipping visualization setup")
            return

        try:
            from metasim.utils.hf_util import check_and_download_recursive
            from metasim.utils.rerun.rerun_util import RerunVisualizer
            from metasim.utils.state import state_tensor_to_nested

            # Download URDF files
            self._download_urdf_files()

            # Initialize visualizer
            self.visualizer = RerunVisualizer(app_name=app_name, spawn=spawn)

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
                    self.visualizer.visualize_item(obj, obj.name, obj_state)

                # Visualize robots
                for robot in self.handler.robots:
                    robot_state = self._extract_state(robot_states.get(robot.name, {}))
                    self.visualizer.visualize_item(robot, robot.name, robot_state)

            print(f"[TaskRerunWrapper] Rerun visualization initialized")

        except Exception as e:
            print(f"[TaskRerunWrapper] Failed to setup visualization: {e}")
            import traceback
            traceback.print_exc()
            self.visualizer = None

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

    def _update_visualization(self):
        """Update visualization with current state."""
        if self.visualizer is None or self.handler is None:
            return

        try:
            from metasim.utils.state import state_tensor_to_nested

            obs = self.handler.get_states(mode="tensor")
            env_states = state_tensor_to_nested(self.handler, obs)

            if env_states and len(env_states) > 0:
                state = env_states[0]
                object_states = state.get("objects", {})
                robot_states = state.get("robots", {})

                # Update objects
                for obj in self.handler.objects:
                    if obj.name in object_states:
                        obj_state = self._extract_state(object_states[obj.name])
                        self.visualizer.update_item_pose(obj.name, obj_state)

                # Update robots
                for robot in self.handler.robots:
                    if robot.name in robot_states:
                        robot_state = self._extract_state(robot_states[robot.name])
                        self.visualizer.update_item_pose(robot.name, robot_state)

                # Update time step
                self.visualizer.set_time(self.step_count)

        except Exception as e:
            print(f"[TaskRerunWrapper] Visualization update failed: {e}")

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

    def close(self):
        """Close environment and visualizer."""
        if self.visualizer is not None:
            self.visualizer.close()
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name):
        """Proxy all other attributes to the wrapped environment."""
        return getattr(self.env, name)

