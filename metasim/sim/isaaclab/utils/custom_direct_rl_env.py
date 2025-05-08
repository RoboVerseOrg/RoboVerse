## TODO: This code need to be checked carefully when upgrading to new IsaacLab version

from typing import Any, Sequence

import torch

try:
    from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn
    from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
except ModuleNotFoundError:
    from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
    from isaaclab.envs.direct_rl_env import DirectRLEnv


## This env class is almost same as DirectRLEnv
class CustomDirectRLEnv(DirectRLEnv):
    def reset(
        self,
        env_ids: Sequence[int] | None = None,  # ! new argument
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """
        Compared to `DirectRLEnv.reset()`, this function support resetting specific environments
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        elif isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.int64, device=self.device)
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # return observations
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Compared to DirectRLEnv.step(), this function won't automatically reset environments that is successful or timed-out.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # ! Below is the original code from DirectRLEnv.step()
        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self._reset_idx(reset_env_ids)
        #     # update articulation kinematics
        #     self.scene.write_data_to_sim()
        #     self.sim.forward()
        #     # if sensors are added to the scene, make sure we render to reflect changes in reset
        #     if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
        #         self.sim.render()
        # ! End of original code

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = None  # XXX: modified to speed up

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def get_observations(self):
        return self._get_observations()

    def get_dones(self):
        return self._get_dones()
