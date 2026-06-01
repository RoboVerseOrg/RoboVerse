"""Fully robosuite-free Lift task on MetaSim's MujocoHandler.

This module imports **no robosuite**. It loads a vendored, standalone Lift MJCF
(see ``tools/robosuite_integration/vendor_assets.py``) into MetaSim's
groundless handler and reproduces robosuite's Lift end-to-end natively:

* control  — :class:`._osc.NativeOSC` (OSC_POSE arm + parallel-jaw gripper)
* placement — native ``UniformRandomSampler`` port (cube xy ∈ [-0.03, 0.03], z-rot)
* reward    — robosuite Lift reward ported (reach + grasp + 2.25 success)
* success   — cube center > table top (0.8) + 0.04 m
* observation — robomimic low-dim keys (proprio + cube pose + gripper-to-cube)

Constants below (Panda rest pose, base frame, table height) are plain numbers
lifted from robosuite once — not a runtime dependency. With the vendored bundle
present, ``robosuite`` can be uninstalled and this task still runs (proven by
``tools/robosuite_integration/verify_robosuite_free.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from metasim.scenario.scenario import ScenarioCfg  # noqa: E402
from metasim.scenario.scene import SceneCfg  # noqa: E402
from metasim.scenario.simulator_params import SimParamCfg  # noqa: E402

from ._osc import NativeOSC  # noqa: E402
from .robosuite_env import _GroundlessMujocoHandler  # noqa: E402

# --- robosuite-derived constants (plain numbers, no runtime dep) ---
_PANDA_INIT_QPOS = np.array([0.0, 0.19635, 0.0, -2.61799, 0.0, 2.94159, 0.7854])
_BASE_POS = np.array([-0.56, 0.0, 0.912])
_BASE_ORI = np.eye(3)
_TABLE_TOP_Z = 0.8
_CUBE_Z = 0.831  # table top + cube half + z_offset
_XY_RANGE = 0.03
_DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / ".robosuite_bundle" / "lift"


def _mat2quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion [x,y,z,w] (robosuite convention)."""
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[3] = (m[k, j] - m[j, k]) / s
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    return q


class NativeLiftEnv:
    """Robosuite Lift, reproduced on MetaSim's handler with zero robosuite imports."""

    control_freq = 20
    horizon = 200

    def __init__(self, bundle_dir: str | Path | None = None, seed: int = 0, reward_shaping: bool = True):
        bundle = Path(bundle_dir or _DEFAULT_BUNDLE)
        model_path = bundle / "lift.xml"
        if not model_path.exists():
            raise FileNotFoundError(
                f"vendored Lift bundle not found at {model_path}. Run once (with robosuite installed):\n"
                "  python -m tools.robosuite_integration.vendor_assets --env Lift --out .robosuite_bundle/lift"
            )
        self._rng = np.random.RandomState(seed)
        self.reward_shaping = reward_shaping
        self.decimation = int(round((1.0 / self.control_freq) / 0.002))

        sc = ScenarioCfg(
            scene=SceneCfg(mjcf_path=str(model_path)),
            robots=[], lights=[], ground=None,
            sim_params=SimParamCfg(dt=0.002), decimation=self.decimation,
            simulator="mujoco", num_envs=1, headless=True,
        )
        self.handler = _GroundlessMujocoHandler(sc)
        self.handler.launch()
        self.m, self.d = self.handler.physics.model.ptr, self.handler.physics.data.ptr

        nid = lambda t, n: mujoco.mj_name2id(self.m, t, n)  # noqa: E731
        self._arm_qpos = [self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_qvel = [self.m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_act = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot0_torq_j{i}") for i in range(1, 8)]
        self._grip_act = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"gripper0_right_gripper_finger_joint{i}") for i in (1, 2)]
        self._grip_qpos = [
            self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, jn)]
            for jn in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
        ]
        self._eef_sid = nid(mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
        self._cube_bid = nid(mujoco.mjtObj.mjOBJ_BODY, "cube_main")
        self._cube_qadr = self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "cube_joint0")]
        # geoms for the contact-based grasp check
        self._finger_geoms = self._geoms_under("gripper0")
        self._cube_geoms = self._geoms_of_body(self._cube_bid)
        self.action_dim = 7

    # ---- model introspection ----
    def _geoms_of_body(self, bid: int) -> set[int]:
        return {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == bid}

    def _geoms_under(self, prefix: str) -> set[int]:
        out = set()
        for g in range(self.m.ngeom):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm.startswith(prefix) and "finger" in nm:
                out.add(g)
        return out

    # ---- placement (native UniformRandomSampler) ----
    def _sample_cube(self) -> np.ndarray:
        x = self._rng.uniform(-_XY_RANGE, _XY_RANGE)
        y = self._rng.uniform(-_XY_RANGE, _XY_RANGE)
        ang = self._rng.uniform(0, 2 * np.pi)
        quat = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])  # z-rotation, [w,x,y,z]
        return np.array([x, y, _CUBE_Z, *quat])

    # ---- env api ----
    def reset(self):
        with self.handler.physics.reset_context():
            self.d.qpos[self._arm_qpos] = _PANDA_INIT_QPOS
            self.d.qpos[self._grip_qpos] = [0.04, -0.04]  # open
            self.d.qpos[self._cube_qadr : self._cube_qadr + 7] = self._sample_cube()
            self.d.qvel[:] = 0.0
        self._osc = NativeOSC(
            self.m, self.d, eef_site="gripper0_right_grip_site",
            arm_joint_qpos=self._arm_qpos, arm_joint_qvel=self._arm_qvel,
            arm_actuator_ids=self._arm_act, gripper_actuator_ids=self._grip_act,
            initial_joint=_PANDA_INIT_QPOS, base_pos=_BASE_POS, base_ori=_BASE_ORI,
        )
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).flatten()
        self._osc.apply(action)
        for _ in range(self.decimation):
            self._osc.write_ctrl()
            mujoco.mj_step(self.m, self.d)
        self._t += 1
        obs = self._obs()
        r = self._reward()
        done = self._t >= self.horizon
        return obs, r, done, {"success": self._check_success()}

    # ---- task logic (ported from robosuite Lift) ----
    def _eef_pos(self) -> np.ndarray:
        return self.d.site_xpos[self._eef_sid].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.d.xpos[self._cube_bid].copy()

    def _check_success(self) -> bool:
        return bool(self._cube_pos()[2] > _TABLE_TOP_Z + 0.04)

    def _check_grasp(self) -> bool:
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in self._finger_geoms and b in self._cube_geoms) or (b in self._finger_geoms and a in self._cube_geoms):
                return True
        return False

    def _reward(self) -> float:
        if self._check_success():
            reward = 2.25
        elif self.reward_shaping:
            dist = float(np.linalg.norm(self._eef_pos() - self._cube_pos()))
            reward = 1.0 - np.tanh(10.0 * dist)
            if self._check_grasp():
                reward += 0.25
        else:
            reward = 0.0
        return reward

    def _obs(self) -> np.ndarray:
        qpos = self.d.qpos[self._arm_qpos]
        qvel = self.d.qvel[self._arm_qvel]
        eef_pos = self._eef_pos()
        eef_quat = _mat2quat(self.d.site_xmat[self._eef_sid].reshape(3, 3))
        grip_qpos = self.d.qpos[self._grip_qpos]
        cube_pos = self._cube_pos()
        cube_quat = self.d.xquat[self._cube_bid].copy()  # [w,x,y,z]
        return np.concatenate(
            [np.cos(qpos), np.sin(qpos), qvel, eef_pos, eef_quat, grip_qpos, cube_pos, cube_quat, cube_pos - eef_pos]
        ).astype(np.float32)

    def close(self):
        self.handler.close()
