"""Native MetaSim LIBERO tasks — bitwise 1:1, runnable with libero uninstalled.

A LIBERO task as a first-class MetaSim ``BaseTaskEnv``: it loads LIBERO's *own*
compiled ``model_file`` (robosuite MJCF, vendored per task) onto MetaSim's
groundless ``MujocoHandler`` (physics bit-for-bit identical to LIBERO — see
``tools/libero_integration``), drives it with the native robosuite-free OSC
controller, judges success natively from the BDDL goal, and exposes the
robomimic-style low-dim observation. No ``import libero`` / ``import robosuite``
at runtime.

A bundle dir (``native_bundles/<task>/``) holds the static, libero-free inputs:
``model.xml`` (the demo's embedded MJCF), ``goal.json`` (parsed BDDL ``:goal``),
``init.npz`` (a demo's initial qpos/qvel). Build them with
``tools/libero_integration/vendor_native_libero.py``. Mesh assets are data
(LIBERO + robosuite trees, like roboverse_data); point ``LIBERO_ASSETS`` at them.

### Title
LIBERO (native, all 130)

### Platforms
- mujoco

### Description
All 130 LIBERO benchmark tasks as first-class MetaSim tasks (``libero_native.*``),
bitwise 1:1 with LIBERO and runnable with libero/robosuite uninstalled.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from roboverse_pack.tasks.robosuite._osc import NativeOSC
from roboverse_pack.tasks.robosuite.robosuite_env import _GroundlessMujocoHandler

from ._native_util import check_bddl_success, remap_libero_model, roboverse_data_assets

_PANDA_INIT = np.array([0.0, 0.19635, 0.0, -2.61799, 0.0, 2.94159, 0.7854])
_BUNDLES = Path(__file__).resolve().parent / "native_bundles"


def _libero_assets() -> str:
    # vendored roboverse_data tree by default (no local LIBERO checkout needed);
    # LIBERO_ASSETS overrides to a local LIBERO ``.../libero/assets`` dir.
    return os.environ.get("LIBERO_ASSETS") or os.path.join(roboverse_data_assets(), "libero")


class NativeLiberoEnv(BaseTaskEnv):
    """LIBERO task on MetaSim's MujocoHandler — native control + BDDL success."""

    bundle: str = ""  # <bundles_root>/<bundle>
    bundles_root: Path = _BUNDLES
    max_episode_steps = 600
    control_freq = 20

    def _remap_model(self, model_xml: str) -> str:
        """Rebase the bundle's embedded MJCF to local assets (overridable per family)."""
        return remap_libero_model(model_xml, _libero_assets())

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None):
        bdir = self.bundles_root / self.bundle
        model_xml = (bdir / "model.xml").read_text()
        self._goal = [tuple(t) for t in json.loads((bdir / "goal.json").read_text())]
        init = np.load(bdir / "init.npz")
        self._qpos0, self._qvel0 = init["qpos"], init["qvel"]

        xml = self._remap_model(model_xml)
        self._xml_file = tempfile.NamedTemporaryFile(suffix=f"_{self.bundle}.xml", delete=False, mode="w")
        self._xml_file.write(xml)
        self._xml_file.flush()
        self._decimation = round((1.0 / self.control_freq) / 0.002)
        sc = ScenarioCfg(
            scene=SceneCfg(mjcf_path=self._xml_file.name),
            robots=[],
            lights=[],
            ground=None,
            sim_params=SimParamCfg(dt=0.002),
            decimation=self._decimation,
            simulator="mujoco",
            num_envs=1,
            headless=True,
        )
        # the remapped MJCF already points at local LIBERO/robosuite mesh assets;
        # skip the roboverse_data HF asset-download check (assets are on disk).
        sc.check_assets = lambda *a, **k: None
        handler = _GroundlessMujocoHandler(sc)
        handler.launch()
        super().__init__(scenario=handler, device=device)
        self._setup_native()

    # ---- native control / state plumbing ----
    @property
    def _m(self):
        return self.handler.physics.model.ptr

    @property
    def _d(self):
        return self.handler.physics.data.ptr

    def _setup_native(self):
        m = self._m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        self._arm_qpos = [m.jnt_qposadr[jid(f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_qvel = [m.jnt_dofadr[jid(f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_act = [aid(f"robot0_torq_j{i}") for i in range(1, 8)]
        self._grip_act = [
            a for a in (aid("gripper0_gripper_finger_joint1"), aid("gripper0_gripper_finger_joint2")) if a >= 0
        ]
        self._eef_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        mujoco.mj_forward(m, self._d)
        self._base_pos = self._d.xpos[bid].copy()
        self._base_ori = self._d.xmat[bid].reshape(3, 3).copy()
        self._action_dim = 7

    def _make_osc(self):
        self._osc = NativeOSC(
            self._m,
            self._d,
            eef_site="gripper0_grip_site",
            arm_joint_qpos=self._arm_qpos,
            arm_joint_qvel=self._arm_qvel,
            arm_actuator_ids=self._arm_act,
            gripper_actuator_ids=self._grip_act,
            initial_joint=_PANDA_INIT,
            base_pos=self._base_pos,
            base_ori=self._base_ori,
        )

    # ---- MetaSim task API ----
    def _observation_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(self._obs_vec().shape[-1],), dtype=np.float32)

    def _action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def _obs_vec(self) -> np.ndarray:
        d = self._d
        qpos = d.qpos[self._arm_qpos]
        eef = d.site_xpos[self._eef_sid]
        return np.concatenate([np.cos(qpos), np.sin(qpos), d.qvel[self._arm_qvel], eef]).astype(np.float32)

    def _observation(self, env_states=None):
        return torch.from_numpy(self._obs_vec()).to(self.device).unsqueeze(0)

    def _terminated(self, env_states=None) -> torch.Tensor:
        ok = check_bddl_success(self._m, self._d, self._goal)
        return torch.tensor([ok], dtype=torch.bool, device=self.device)

    def _get_initial_states(self):
        return None

    def reset(self, states=None, env_ids=None):
        nq, nv = self._m.nq, self._m.nv
        with self.handler.physics.reset_context():
            self._d.qpos[:] = self._qpos0[:nq]
            self._d.qvel[:] = self._qvel0[:nv]
        self._make_osc()
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return self._observation(), {}

    def step(self, actions):
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions, dtype=float)
        a = a.flatten()[: self._action_dim]
        self._osc.apply(a)
        for _ in range(self._decimation):
            self._osc.write_ctrl()
            mujoco.mj_step(self._m, self._d)
        self._episode_steps = self._episode_steps + 1
        success = check_bddl_success(self._m, self._d, self._goal)
        timeout = self._episode_steps >= self.max_episode_steps
        terminated = torch.tensor([success], dtype=torch.bool, device=self.device)
        reward = torch.tensor([1.0 if success else 0.0], dtype=torch.float32, device=self.device)
        return self._observation(), reward, terminated, timeout, {"success": success}

    def set_full_state(self, qpos, qvel):
        with self.handler.physics.reset_context():
            self._d.qpos[:] = qpos
            self._d.qvel[:] = qvel
        self._make_osc()
        return self._observation()

    def close(self):
        super().close()
        # remove the per-instance temp MJCF (written with delete=False so the
        # handler could open it by path); otherwise these leak into the tmp dir.
        f = getattr(self, "_xml_file", None)
        if f is not None:
            try:
                os.unlink(f.name)
            except OSError:
                pass
            self._xml_file = None


def _register_bundled_tasks():
    """Register a NativeLiberoEnv subclass per vendored bundle: ``libero_native.<bundle>``."""
    if not _BUNDLES.exists():
        return
    for bdir in sorted(_BUNDLES.iterdir()):
        if not (bdir / "model.xml").exists():
            continue
        name = bdir.name
        cls = type(f"NativeLibero_{name}", (NativeLiberoEnv,), {"bundle": name, "__doc__": f"LIBERO {name} (native)."})
        register_task(f"libero_native.{name}")(cls)


_register_bundled_tasks()
