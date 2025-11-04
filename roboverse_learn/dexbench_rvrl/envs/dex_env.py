from __future__ import annotations

import copy
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.types import DictEnvState
from metasim.utils.setup_util import get_handler
from roboverse_pack.tasks.dexbench.get_task import get_task
from metasim.utils.state import list_state_to_tensor
from roboverse_learn.dexbench_rvrl.envs.base import BaseVecEnv
from .domain_randomization_helper import DomainRandomizationHelper


class DexEnv(BaseVecEnv):
    """Dex Environment Wrapper for RL tasks."""

    def __init__(self, task_name: str, args=None, env_cfg=None):
        """Initialize the BiDex environment wrapper."""
        assert args is not None, "args must be provided"
        task_cls = get_task(task_name)
        task = task_cls()
        if args.objects is not None:
            task.current_object_type = args.objects
        if args.robot is not None:
            task.current_robot_type = args.robot
        task.num_envs = args.num_envs
        if args.sim_device is None:
            args.sim_device = args.device
        task.device = args.sim_device
        task.is_testing = args.test
        task.obs_type = args.obs_type
        if "rgb" in args.obs_type and env_cfg is not None:
            task.img_h = env_cfg.get("img_h", 256)
            task.img_w = env_cfg.get("img_w", 256)
        task.use_prio = not args.no_prio  # Use proprioception in state observation
        task.set_sim_params(sim_type=args.sim)
        task.set_objects()
        task.set_init_states()
        self.task = task
        cameras = [] if not hasattr(task, "cameras") else task.cameras
        sensors = [] if not hasattr(task, "sensors") else task.sensors
        light_cfg = env_cfg.get("light", None) if env_cfg is not None else None
        if light_cfg is not None:
            from metasim.scenario.lights import BaseLightCfg, DistantLightCfg, CylinderLightCfg, DomeLightCfg, SphereLightCfg, DiskLightCfg

            light_type = light_cfg.get("type", "distant").lower()
            if light_type == "distant":
                light = DistantLightCfg(
                    name="distant_light",
                    intensity=light_cfg.get("intensity", 500.0),
                    color=tuple(light_cfg.get("color", (1.0, 1.0, 1.0))),
                    polar=light_cfg.get("polar", 0.0),
                    azimuth=light_cfg.get("azimuth", 0.0),
                )
            elif light_type == "cylinder":
                light = CylinderLightCfg(
                    name="cylinder_light",
                    intensity=light_cfg.get("intensity", 500.0),
                    color=tuple(light_cfg.get("color", (1.0, 1.0, 1.0))),
                    length=light_cfg.get("length", 1.0),
                    radius=light_cfg.get("radius", 0.5),
                )
            elif light_type == "dome":
                light = DomeLightCfg(
                    name="dome_light",
                    intensity=light_cfg.get("intensity", 500.0),
                    color=tuple(light_cfg.get("color", (1.0, 1.0, 1.0))),
                )
            elif light_type == "sphere":
                light = SphereLightCfg(
                    name="sphere_light",
                    intensity=light_cfg.get("intensity", 500.0),
                    color=tuple(light_cfg.get("color", (1.0, 1.0, 1.0))),
                    radius=light_cfg.get("radius", 0.5),    
                )
            elif light_type == "disk":
                light = DiskLightCfg(
                    name="disk_light",
                    intensity=light_cfg.get("intensity", 500.0),
                    color=tuple(light_cfg.get("color", (1.0, 1.0, 1.0))),
                    radius=light_cfg.get("radius", 1.0),
                )
            else:
                log.warning(f"Unknown light type '{light_type}', defaulting to DistantLightCfg.")
                light = DistantLightCfg()
        else:
            from metasim.scenario.lights import DistantLightCfg
            light = DistantLightCfg()
                
        scenario = ScenarioCfg(
            robots=task.robots,
            objects=task.objects,
            sensors=sensors,
            cameras=cameras,
            simulator=args.sim,
            headless=args.headless,
            num_envs=args.num_envs,
            sim_params=task.sim_params,
            decimation=task.decimation,
            env_spacing=task.env_spacing,
            lights=[light], 
        )
        scenario.device = args.sim_device
        self.task.robots = scenario.robots
        self.task.objects = scenario.objects
        self.sim_device = torch.device(args.sim_device if torch.cuda.is_available() else "cpu")
        self._num_envs = scenario.num_envs
        self.robots = scenario.robots
        self.objects = scenario.objects

        self.handler = get_handler(scenario)
        
        randomization_cfg = task.randomization_cfg if hasattr(task, "randomization_cfg") else None
        self.domain_randomization_helper = None
        self.use_dr = args.use_dr
        self.use_materials = args.use_materials
        if randomization_cfg is not None and (args.use_dr or args.use_materials):
            self.domain_randomization_helper = DomainRandomizationHelper(
                cfg=randomization_cfg,
                num_envs=self._num_envs,
                lights=scenario.lights,
                objects=self.objects,
                robots=self.robots,
                cameras=scenario.cameras,
                handler=self.handler,
                env_spacing=scenario.env_spacing,
                seed=args.seed,
                device=self.sim_device,
            )
        self.global_step_count = 0
            

        self.init_states = [copy.deepcopy(self.task.init_states) for _ in range(self.num_envs)]
        self.init_states_tensor = list_state_to_tensor(
            handler=self.handler, env_states=self.init_states, device=self.sim_device
        )

        # action space is normalized to [-1, 1]
        self.action_shape = self.task.action_shape
        self.num_joints = 0
        for robot in self.robots:
            self.num_joints += robot.num_joints
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_shape,), dtype=np.float32)

        # observation space
        # Create an observation space (398 dimensions) for a single environment, instead of the entire batch (num_envs,398).
        self.obs_type = getattr(self.task, "obs_type", "state")
        obs_shape = self.task.obs_shape
        self._observation_space = spaces.Dict({
            key: spaces.Box(low=-5.0, high=5.0, shape=shape, dtype=np.float32) for key, shape in obs_shape.items()
        })
        if hasattr(self.task, "proprio_shape"):
            self.proprio_shape = self.task.proprio_shape
        else:
            self.proprio_shape = None
        if hasattr(self.task, "img_h") and hasattr(self.task, "img_w"):
            self.img_h = self.task.img_h
            self.img_w = self.task.img_w
        else:
            self.img_h = None
            self.img_w = None
        self.tensor_states = None
        self.tensor_obs = None
        log.info(f"Observation space: {self.observation_space}")
        log.info(f"Action space: {self.action_space}")

        self.max_episode_steps = self.task.episode_length
        log.info(f"Max episode steps: {self.max_episode_steps}")

        # Episode tracking variables for EpisodeLogCallback
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.total_reset = 0
        self.total_success = 0
        self.mean_success_rate = 0.0
        self.quelen = 25600 // self.num_envs
        self.reset_counts = deque(maxlen=self.quelen)
        self.success_counts = deque(maxlen=self.quelen)

        self.last_success_rate = 0.0
        if args.seed is not None:
            self.seed(args.seed)

    def scale_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to the range of the action space.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        return self.task.scale_action_fn(
            actions=actions,
        )

    def reset(self):
        """Reset the environment."""
        env_ids = list(range(self.num_envs))
        self.handler.set_states(self.init_states_tensor, env_ids=env_ids)
        if self.domain_randomization_helper is not None and (self.use_dr or self.use_materials):
            self.domain_randomization_helper.randomization(env_ids=env_ids, step_count=self.global_step_count)
        self.handler.refresh_render()
        obs = self.handler.get_states()
        self.task.update_state(obs)
        env_ids = torch.arange(self.num_envs, device=self.sim_device)
        self.tensor_states = obs
        observations = self.task.observation_fn(
            obs, torch.zeros((self.num_envs, self.action_shape), device=self.sim_device)
        )
        self.tensor_obs = {
            key: value.clone() for key, value in observations.items()
        }
        for key, value in observations.items():
            value = torch.clamp(
                value,
                torch.tensor(self.observation_space[key].low, device=self.sim_device),
                torch.tensor(self.observation_space[key].high, device=self.sim_device),
            )
        self.reset_goal_pose(env_ids)

        # Reset episode tracking variables
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)

        log.info("reset now")
        return observations

    def pre_physics_step(self, actions: torch.Tensor):
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        step_action = self.scale_action(actions)
        return step_action

    def post_physics_step(self, envstates: list[DictEnvState], actions: torch.Tensor):
        """Post physics step processing."""
        self.episode_lengths += 1
        (self.episode_rewards, self.episode_reset, self.episode_goal_reset, self.episode_success) = self.task.reward_fn(
            envstates=envstates,
            actions=actions,
            reset_buf=self.episode_reset,
            reset_goal_buf=self.episode_goal_reset,
            episode_length_buf=self.episode_lengths,
            success_buf=self.episode_success,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).

        Returns:
            tuple: A tuple containing the following elements:
                - observations (torch.Tensor, shape=(num_envs, obs_dim)): Observations from the environment.
                - rewards (torch.Tensor, shape=(num_envs,)): Reward values for each environment.
                - dones (torch.Tensor, shape=(num_envs,)): Flags indicating if the episode has ended for each environment (due to termination or truncation).
                - infos (list[dict]): List of additional information for each environment. Each dictionary contains the "TimeLimit.truncated" key,
                                      indicating if the episode was truncated due to timeout.
        """
        actions = torch.clamp(actions, -1.0, 1.0)  # Ensure actions are within [-1, 1]
        step_action = self.pre_physics_step(actions)
        self.handler.set_dof_targets(step_action)
        self.handler.simulate()
        envstates = self.handler.get_states()
        self.task.update_state(envstates)
        self.tensor_obs = self.task.observation_fn(envstates=envstates, actions=actions, device=self.sim_device)
        self.post_physics_step(envstates, actions)

        rewards = deepcopy(self.episode_rewards)
        dones = deepcopy(self.episode_reset)
        info = {}
        info["successes"] = deepcopy(self.episode_success)

        step_resets = self.episode_reset.sum().item()
        step_successes = torch.logical_and(self.episode_reset.bool(), self.episode_success.bool()).sum().item()

        success_rate = step_successes / step_resets if step_resets else self.last_success_rate
        self.last_success_rate = success_rate
        if len(self.reset_counts) == self.quelen:
            self.total_reset -= self.reset_counts[0]
            self.total_success -= self.success_counts[0]

        self.reset_counts.append(step_resets)
        self.success_counts.append(step_successes)

        self.total_reset += step_resets
        self.total_success += step_successes

        if self.total_reset > 0:
            self.mean_success_rate = self.total_success / self.total_reset

        env_ids = self.episode_reset.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.episode_goal_reset.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_goal_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_goal_pose(goal_env_ids)

        if len(env_ids) > 0:
            envstates = self.reset_env(env_ids)

        self.task.update_state(envstates)
        self.tensor_states = envstates
        observations = self.task.observation_fn(envstates=envstates, actions=actions, device=self.sim_device)
        for key, value in observations.items():
            value = torch.clamp(
                value,
                torch.tensor(self.observation_space[key].low, device=self.sim_device),
                torch.tensor(self.observation_space[key].high, device=self.sim_device),
            )

        info["success_rate"] = torch.tensor([success_rate], dtype=torch.float32, device=self.sim_device)
        info["total_succ_rate"] = torch.tensor([self.mean_success_rate], dtype=torch.float32, device=self.sim_device)
        
        self.global_step_count += 1

        return observations, rewards, dones, dones, info

    def reset_env(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        self.episode_lengths[env_ids] = 0
        self.episode_rewards[env_ids] = 0.0
        self.episode_success[env_ids] = 0
        self.episode_reset[env_ids] = 0
        reset_states = self.task.reset_init_pose_fn(self.init_states_tensor, env_ids=env_ids)
        self.reset_goal_pose(env_ids)
        self.handler.set_states(reset_states, env_ids=env_ids.tolist())
        if self.domain_randomization_helper is not None and self.use_dr:
            self.domain_randomization_helper.randomization(env_ids=env_ids.tolist(), step_count=self.global_step_count)
        self.handler.refresh_render()
        env_states = self.handler.get_states()
        return env_states

    def reset_goal_pose(self, env_ids: torch.Tensor):
        """Reset the goal pose for specific environments."""
        self.episode_goal_reset[env_ids] = 0
        if self.task.goal_reset_fn is not None:
            self.task.goal_reset_fn(env_ids=env_ids)
        else:
            log.warning("No goal reset function defined in the task. Skipping goal reset.")

    def close(self):
        """Clean up environment resources."""
        self.handler.close()

    def seed(self, seed):
        """Set random seed for reproducibility."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if hasattr(self.handler, "seed"):
            self.handler.seed(seed)
        elif hasattr(self.handler, "set_seed"):
            self.handler.set_seed(seed)
        else:
            log.warning("Could not set seed on underlying handler.")
        return seed

    def render(self):
        raise NotImplementedError

    @property
    def single_observation_space(self):
        return self._observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def single_action_space(self):
        return self._action_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return self._num_envs
