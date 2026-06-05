"""Native robosuite → MetaSim task: physics on ``MujocoHandler``, logic from robosuite.

Design (see ``tools/robosuite_integration`` for the parity proof):

* robosuite's *own* compiled MJCF is loaded as the MetaSim scene and stepped on
  MetaSim's ``MujocoHandler``. We verified this reproduces robosuite's native
  ``MjSim`` bit-for-bit (qpos/qvel max-abs diff 0.0) given identical actuator
  commands, because both use the same ``mujoco`` 3.x binding and the same XML.

* A native robosuite env is held as the **task oracle**: it owns the OSC
  controller (high-level action → joint torques), the reward, the observation
  builder and the success check. Each step we (1) sync MetaSim's state into the
  oracle, (2) let the oracle's controller emit the per-substep torques for the
  action, (3) apply those torques on the MetaSim handler, (4) read reward /
  obs / success from the oracle re-synced to MetaSim's resulting state.

The handler is the source of truth for physics; the oracle is the source of
truth for control + task semantics. Because the two stay bit-identical, a
policy's return and success rate on this env equal robosuite's exactly.

robosuite mutates the *compiled* model's ``body_pos`` for welded objects (e.g.
the Door) at reset — ``get_xml()`` omits that — so we re-apply ``body_pos`` /
``body_quat`` to the handler model on every reset. Free-joint objects place via
``qpos`` and ride along with the state sync.
"""

from __future__ import annotations

import logging
import os
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import gymnasium as gym
import mujoco
import numpy as np
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.sim.mujoco.mujoco import MujocoHandler
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

_MODEL_TIMESTEP = 0.002  # robosuite macros.SIMULATION_TIMESTEP


class _GroundlessMujocoHandler(MujocoHandler):
    """MujocoHandler that does NOT inject a default ground / lights.

    robosuite's MJCF already supplies its arena floor, lights and textures;
    MetaSim's default ground would duplicate the ``texplane`` texture (hard
    error) and add a second floor plane that changes contact dynamics.
    """

    def _add_ground(self, mjcf_model) -> None:
        return None


class RobosuiteEnv(BaseTaskEnv):
    """Base class for robosuite tasks ported onto MetaSim's MujocoHandler."""

    # ----- subclass config -----
    env_name: str = ""  # robosuite env id, e.g. "Lift"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 20
    max_episode_steps: int = 200
    traj_filepath = None

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None, seed: int = 0):
        self._seed = seed
        self._oracle = self._make_oracle()
        np.random.seed(seed)
        self._oracle.reset()

        self._decimation = round((1.0 / self.control_freq) / _MODEL_TIMESTEP)
        # capture per-substep ctrl emitted by robosuite's controller
        self._substep_ctrl: list[np.ndarray] = []
        self._wrap_pre_action()

        # export the compiled MJCF and build a groundless handler around it
        xml = self._oracle.model.get_xml()
        self._xml_file = tempfile.NamedTemporaryFile(suffix=f"_{self.env_name}.xml", delete=False, mode="w")
        self._xml_file.write(xml)
        self._xml_file.flush()
        handler_scenario = ScenarioCfg(
            scene=SceneCfg(mjcf_path=self._xml_file.name),
            robots=[],
            lights=[],
            ground=None,
            sim_params=SimParamCfg(dt=_MODEL_TIMESTEP),
            decimation=self._decimation,
            simulator="mujoco",
            num_envs=1,
            headless=True,
        )
        handler = _GroundlessMujocoHandler(handler_scenario)
        handler.launch()
        super().__init__(scenario=handler, device=device)

        self._action_dim = int(self._oracle.action_dim)
        self._sync_oracle_to_handler()  # place handler at the oracle's reset state

    # ------------------------------------------------------------------ oracle
    def _make_oracle(self):
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        cfg = load_composite_controller_config(controller=self.controller, robot=self.robot)
        return robosuite.make(
            self.env_name,
            robots=self.robot,
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=self.control_freq,
            horizon=self.max_episode_steps,
            ignore_done=True,
            hard_reset=False,  # keep model topology stable across resets
        )

    def _wrap_pre_action(self) -> None:
        oracle = self._oracle
        orig = oracle._pre_action
        data = oracle.sim.data._data

        def patched(action, policy_step=False):
            orig(action, policy_step)
            self._substep_ctrl.append(data.ctrl.copy())

        oracle._pre_action = patched

    # ------------------------------------------------------------- model sync
    @property
    def _hm(self):  # handler mujoco model
        return self.handler.physics.model.ptr

    @property
    def _hd(self):  # handler mujoco data
        return self.handler.physics.data.ptr

    @property
    def _om(self):
        return self._oracle.sim.model._model

    @property
    def _od(self):
        return self._oracle.sim.data._data

    def _apply_model_overrides(self) -> None:
        """Copy compiled-model placement fields robosuite mutated at reset."""
        self._hm.body_pos[:] = self._om.body_pos
        self._hm.body_quat[:] = self._om.body_quat

    def _sync_oracle_to_handler(self) -> None:
        """Place the handler's physics at the oracle's current qpos/qvel."""
        self._apply_model_overrides()
        physics = self.handler.physics
        with physics.reset_context():
            self._hd.qpos[:] = self._od.qpos
            self._hd.qvel[:] = self._od.qvel

    def _sync_handler_to_oracle(self) -> None:
        """Place the oracle's sim at the handler's current qpos/qvel and forward."""
        self._od.qpos[:] = self._hd.qpos
        self._od.qvel[:] = self._hd.qvel
        self._oracle.sim.forward()

    # --------------------------------------------------------------- task api
    def _observation_space(self) -> gym.Space:
        dim = self._obs_vector(self._oracle._get_observations()).shape[-1]
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(int(self._oracle.action_dim),), dtype=np.float32)

    @staticmethod
    def _obs_vector(obs_dict: dict) -> np.ndarray:
        """Flatten robosuite obs to the robomimic-style low-dim vector."""
        parts = []
        for key in ("robot0_proprio-state", "object-state"):
            if key in obs_dict:
                parts.append(np.asarray(obs_dict[key], dtype=np.float32).ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _to_obs_tensor(self, obs_dict: dict) -> torch.Tensor:
        return torch.from_numpy(self._obs_vector(obs_dict)).to(self.device).unsqueeze(0)

    def reset(self, states=None, env_ids=None):
        np.random.seed(self._seed)
        obs_dict = self._oracle.reset()
        self._sync_oracle_to_handler()
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return self._to_obs_tensor(obs_dict), {}

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            a = actions.detach().cpu().numpy()
        else:
            a = np.asarray(actions, dtype=np.float64)
        a = a.flatten()[: self._action_dim]

        # 1) oracle sees the handler's current state, emits per-substep torques
        self._sync_handler_to_oracle()
        self._substep_ctrl.clear()
        self._oracle.step(a)
        ctrls = self._substep_ctrl

        # 2) apply the same torques on the MetaSim handler
        hd, hm = self._hd, self._hm
        for c in ctrls:
            hd.ctrl[:] = c
            mujoco.mj_step(hm, hd)

        # 3) task signals from robosuite, evaluated on the handler's resulting state
        self._sync_handler_to_oracle()
        obs_dict = self._oracle._get_observations(force_update=True)
        reward = float(self._oracle.reward(a))
        success = bool(self._oracle._check_success())

        self._episode_steps = self._episode_steps + 1
        timeout = self._episode_steps >= self.max_episode_steps
        terminated = torch.full((self.num_envs,), success, dtype=torch.bool, device=self.device)
        reward_t = torch.full((self.num_envs,), reward, dtype=torch.float32, device=self.device)
        return self._to_obs_tensor(obs_dict), reward_t, terminated, timeout, {"success": success}

    def _get_initial_states(self):
        return None

    def close(self) -> None:
        try:
            self._oracle.close()
        except Exception:
            pass
        super().close()
        # remove the per-instance temp MJCF (delete=False so the handler could
        # open it by path); otherwise these leak into the tmp dir.
        f = getattr(self, "_xml_file", None)
        if f is not None:
            try:
                os.unlink(f.name)
            except OSError:
                pass
            self._xml_file = None


# --------------------------------------------------------------------- tasks
@register_task("robosuite.lift", "robosuite_lift")
class LiftEnv(RobosuiteEnv):
    """Lift a cube off the table (robosuite ``Lift``)."""

    env_name = "Lift"
    max_episode_steps = 200


@register_task("robosuite.stack", "robosuite_stack")
class StackEnv(RobosuiteEnv):
    """Stack cube A on cube B (robosuite ``Stack``)."""

    env_name = "Stack"
    max_episode_steps = 200


@register_task("robosuite.door", "robosuite_door")
class DoorEnv(RobosuiteEnv):
    """Turn the handle and open the door (robosuite ``Door``)."""

    env_name = "Door"
    max_episode_steps = 300


@register_task("robosuite.pick_place_can", "robosuite_pick_place_can")
class PickPlaceCanEnv(RobosuiteEnv):
    """Pick the can and place it in its bin (robosuite ``PickPlaceCan``)."""

    env_name = "PickPlaceCan"
    max_episode_steps = 400


@register_task("robosuite.nut_assembly_square", "robosuite_nut_assembly_square")
class NutAssemblySquareEnv(RobosuiteEnv):
    """Fit the square nut on the square peg (robosuite ``NutAssemblySquare``)."""

    env_name = "NutAssemblySquare"
    max_episode_steps = 400
