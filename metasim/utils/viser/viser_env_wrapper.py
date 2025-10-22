from typing import Any

from metasim.utils.state import list_state_to_tensor


class TaskViserWrapper:
    """Simple wrapper for RLTaskEnv with real-time Viser visualization.

    Only renders the first environment for simplicity. Designed for fast_td3 integration.
    """

    def __init__(self, task_env, port: int = 8080, update_freq: int = 1):
        """Initialize the wrapper.

        Args:
            task_env: RLTaskEnv or similar environment with handler
            port: Port for Viser server
            update_freq: Update visualization every N steps to reduce resource usage
        """
        self.env = task_env
        self.visualizer = None
        self.update_freq = update_freq
        self.step_count = 0
        self._setup_visualization(port)

    def _setup_visualization(self, port: int):
        """Setup Viser visualization."""
        try:
            from metasim.utils.viser.viser_util import ViserVisualizer

            self.visualizer = ViserVisualizer(port=port)
            self.visualizer.add_grid()
            self.visualizer.add_frame("/world_frame")

            # Download URDF files for visualization
            try:
                from get_started.viser.viser_demo import download_urdf_files

                download_urdf_files(self.env.scenario)
            except ImportError:
                pass  # Skip if function not available

            # Get initial states using env.handler
            obs = self.env.handler.get_states(mode="tensor")

            # Extract initial states for first environment only using helper method
            if hasattr(obs, "objects") and hasattr(obs, "robots"):
                default_object_states = self._extract_states_from_obs(obs, "objects")
                default_robot_states = self._extract_states_from_obs(obs, "robots")

                # Visualize all objects and robots (like viser_demo.py)
                if hasattr(self.env.scenario, "objects") and self.env.scenario.objects:
                    self.visualizer.visualize_scenario_items(self.env.scenario.objects, default_object_states)

                if hasattr(self.env.scenario, "robots") and self.env.scenario.robots:
                    self.visualizer.visualize_scenario_items(self.env.scenario.robots, default_robot_states)

            # Setup camera
            self.visualizer.enable_camera_controls(
                initial_position=[1.5, -1.5, 1.5],
                render_width=1024,
                render_height=1024,
                look_at_position=[0, 0, 0],
                initial_fov=71.28,
            )

            # Viser visualization successfully initialized

        except ImportError:
            pass  # Silently disable visualization if Viser not available

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
                joint_names = self.env.handler._get_joint_names(name, sort=True)
                state_dict["dof_pos"] = {joint_names[i]: item.joint_pos[0, i].item() for i in range(len(joint_names))}

            result[name] = state_dict

        return result

    def _update_viser_states(self, obs):
        """Update Viser visualization with current states from first environment."""
        if self.visualizer is None or obs is None:
            return

        try:
            # Extract states from first environment using helper method
            if hasattr(obs, "objects") and hasattr(obs, "robots"):
                # Update objects from first environment
                if hasattr(self.env.scenario, "objects"):
                    object_states = self._extract_states_from_obs(obs, "objects")
                    for name, state in object_states.items():
                        self.visualizer.update_item_pose(name, state)

                # Update robots from first environment
                if hasattr(self.env.scenario, "robots"):
                    robot_states = self._extract_states_from_obs(obs, "robots")
                    for name, state in robot_states.items():
                        self.visualizer.update_item_pose(name, state)

            self.visualizer.refresh_camera_view()
        except Exception as e:
            # Silently handle visualization errors to not break training
            pass

    def reset(self, **kwargs):
        """Reset environment and update visualization."""
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) and len(result) > 0 else result

        # Get states using env.handler for consistency
        if hasattr(self.env, "handler") and self.env.handler is not None:
            handler_obs = self.env.handler.get_states(mode="tensor")
            if handler_obs is not None:
                obs = handler_obs

        self._update_viser_states(obs)
        return result

    def set_states(self, states):
        """Set environment states using handler.

        Args:
            states: List of state dictionaries, one per environment, or single state dict
                   Each state dict should have format:
                   {
                       "objects": {obj_name: {"pos": [...], "rot": [...], "dof_pos": {...}}},
                       "robots": {robot_name: {"pos": [...], "rot": [...], "dof_pos": {...}}}
                   }
        """
        if not hasattr(self.env, "handler") or self.env.handler is None:
            return

        # Convert to list format if single state dict provided
        if isinstance(states, dict) and "objects" in states and "robots" in states:
            states = [states] * (self.env.num_envs if hasattr(self.env, "num_envs") else 1)

        # Convert nested state dict to tensor state and set
        tensor_state = list_state_to_tensor(self.env.handler, states)
        self.env.handler.set_states(tensor_state)

    def step(self, actions):
        """Step environment and update visualization."""
        result = self.env.step(actions)
        obs = result[0] if isinstance(result, tuple) and len(result) > 0 else result

        # Get states using env.handler for consistency
        if hasattr(self.env, "handler") and self.env.handler is not None:
            handler_obs = self.env.handler.get_states(mode="tensor")
            if handler_obs is not None:
                obs = handler_obs

        # Update visualization only at specified frequency to reduce resource usage
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self._update_viser_states(obs)

        return result

    def render(self, mode="human"):
        """Render the environment."""
        return None

    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()

    # Delegate other methods to wrapped environment
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)

    def __len__(self) -> int:
        """Return number of environments."""
        return self.env.num_envs if hasattr(self.env, "num_envs") else 1

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self.env.num_envs if hasattr(self.env, "num_envs") else 1

    @property
    def num_actions(self) -> int:
        """Number of actions."""
        return self.env.num_actions if hasattr(self.env, "num_actions") else 0

    @property
    def num_obs(self) -> int:
        """Number of observations."""
        return self.env.num_obs if hasattr(self.env, "num_obs") else 0

    @property
    def max_episode_steps(self) -> int:
        """Maximum episode steps."""
        return self.env.max_episode_steps if hasattr(self.env, "max_episode_steps") else 1000

    @property
    def action_space(self):
        """Action space."""
        return self.env.action_space if hasattr(self.env, "action_space") else None

    @property
    def observation_space(self):
        """Observation space."""
        return self.env.observation_space if hasattr(self.env, "observation_space") else None
