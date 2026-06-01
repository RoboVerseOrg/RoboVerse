"""Fully robosuite-free robosuite tasks on MetaSim's MujocoHandler.

This module imports **no robosuite**. Each task loads a vendored, standalone MJCF
(``tools/robosuite_integration/vendor_assets.py``) into MetaSim's groundless
handler and reproduces robosuite's task natively:

* control      — :class:`._osc.NativeOSC` (OSC_POSE arm + parallel-jaw gripper)
* placement    — native uniform sampler over the free-joint objects
* success      — ported from each robosuite env's ``_check_success``
* reward       — reach + grasp + success bonus (ported, success-faithful)
* observation  — proprio + per-object pose

Constants (Panda rest pose, table height) are plain numbers lifted from robosuite
once — not a runtime dependency. With a vendored bundle present, ``robosuite`` can
be uninstalled and these tasks still run (see ``verify_robosuite_free.py``).
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

_PANDA_INIT_QPOS = np.array([0.0, 0.19635, 0.0, -2.61799, 0.0, 2.94159, 0.7854])
_TABLE_TOP_Z = 0.8
_BUNDLE_ROOT = Path(__file__).resolve().parents[3] / ".robosuite_bundle"


def _mat2quat(m: np.ndarray) -> np.ndarray:
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


class NativeRobosuiteEnv:
    """Base: shared Panda OSC control + handler; subclasses add task semantics."""

    ENV_NAME = ""
    horizon = 200
    control_freq = 20
    # free-joint objects to (re)place at reset: (body_name, free_joint_name, z)
    PLACE_OBJECTS: list[tuple[str, str, float]] = []
    XY_RANGE = 0.03

    def __init__(self, bundle_dir: str | Path | None = None, seed: int = 0):
        bundle = Path(bundle_dir or (_BUNDLE_ROOT / self.ENV_NAME.lower()))
        model_path = bundle / f"{bundle.name}.xml"  # vendored model is named after the bundle dir
        if not model_path.exists():
            raise FileNotFoundError(
                f"vendored bundle missing: {model_path}. Run (robosuite installed):\n"
                f"  python -m tools.robosuite_integration.vendor_assets --env {self.ENV_NAME} "
                f"--out .robosuite_bundle/{self.ENV_NAME.lower()}"
            )
        self._rng = np.random.RandomState(seed)
        self.decimation = int(round((1.0 / self.control_freq) / 0.002))
        sc = ScenarioCfg(
            scene=SceneCfg(mjcf_path=str(model_path)), robots=[], lights=[], ground=None,
            sim_params=SimParamCfg(dt=0.002), decimation=self.decimation,
            simulator="mujoco", num_envs=1, headless=True,
        )
        self.handler = _GroundlessMujocoHandler(sc)
        self.handler.launch()
        self.m, self.d = self.handler.physics.model.ptr, self.handler.physics.data.ptr
        self._setup_common()
        self._setup_task()
        self.action_dim = 7

    # ---- model introspection (no robosuite) ----
    def _jid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def _bid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)

    def _aid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def _sid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, n)

    def _geoms_of_body(self, bid: int) -> set[int]:
        return {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == bid}

    def _setup_common(self):
        self._arm_qpos = [self.m.jnt_qposadr[self._jid(f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_qvel = [self.m.jnt_dofadr[self._jid(f"robot0_joint{i}")] for i in range(1, 8)]
        self._arm_act = [self._aid(f"robot0_torq_j{i}") for i in range(1, 8)]
        self._grip_act = [self._aid(f"gripper0_right_gripper_finger_joint{i}") for i in (1, 2)]
        self._grip_qpos = [self.m.jnt_qposadr[self._jid(f"gripper0_right_finger_joint{i}")] for i in (1, 2)]
        self._eef_sid = self._sid("gripper0_right_grip_site")
        self._finger_geoms = {
            g for g in range(self.m.ngeom)
            if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("gripper0")
            and "finger" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        }
        bid = self._bid("robot0_base")
        mujoco.mj_forward(self.m, self.d)
        self._base_pos = self.d.xpos[bid].copy()
        self._base_ori = self.d.xmat[bid].reshape(3, 3).copy()

    # ---- hooks (subclasses) ----
    def _setup_task(self):
        ...

    def _check_success(self) -> bool:
        return False

    def _task_reward(self) -> float:
        return 2.25 if self._check_success() else 0.0

    def _object_obs(self) -> list[np.ndarray]:
        return []

    # ---- placement ----
    def _place_objects(self):
        for body, joint, z in self.PLACE_OBJECTS:
            adr = self.m.jnt_qposadr[self._jid(joint)]
            x = self._rng.uniform(-self.XY_RANGE, self.XY_RANGE)
            y = self._rng.uniform(-self.XY_RANGE, self.XY_RANGE)
            ang = self._rng.uniform(0, 2 * np.pi)
            self.d.qpos[adr : adr + 7] = [x, y, z, np.cos(ang / 2), 0, 0, np.sin(ang / 2)]

    # ---- env api ----
    def reset(self):
        with self.handler.physics.reset_context():
            self.d.qpos[self._arm_qpos] = _PANDA_INIT_QPOS
            self.d.qpos[self._grip_qpos] = [0.04, -0.04]
            self.d.qvel[:] = 0.0
            self._place_objects()
        self._make_osc()
        self._t = 0
        return self._obs()

    def set_full_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set the entire sim state (e.g. a benchmark demo's initial frame)."""
        with self.handler.physics.reset_context():
            self.d.qpos[:] = qpos
            self.d.qvel[:] = qvel
        self._make_osc()
        self._t = 0
        return self._obs()

    def _make_osc(self):
        self._osc = NativeOSC(
            self.m, self.d, eef_site="gripper0_right_grip_site",
            arm_joint_qpos=self._arm_qpos, arm_joint_qvel=self._arm_qvel,
            arm_actuator_ids=self._arm_act, gripper_actuator_ids=self._grip_act,
            initial_joint=_PANDA_INIT_QPOS, base_pos=self._base_pos, base_ori=self._base_ori,
        )

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).flatten()
        self._osc.apply(action)
        for _ in range(self.decimation):
            self._osc.write_ctrl()
            mujoco.mj_step(self.m, self.d)
        self._t += 1
        return self._obs(), self._task_reward(), self._t >= self.horizon, {"success": self._check_success()}

    # ---- shared accessors ----
    def _eef_pos(self):
        return self.d.site_xpos[self._eef_sid].copy()

    def _body_pos(self, bid):
        return self.d.xpos[bid].copy()

    def _grasping(self, obj_geoms: set[int]) -> bool:
        for i in range(self.d.ncon):
            a, b = self.d.contact[i].geom1, self.d.contact[i].geom2
            if (a in self._finger_geoms and b in obj_geoms) or (b in self._finger_geoms and a in obj_geoms):
                return True
        return False

    def _contact(self, ga: set[int], gb: set[int]) -> bool:
        for i in range(self.d.ncon):
            a, b = self.d.contact[i].geom1, self.d.contact[i].geom2
            if (a in ga and b in gb) or (b in ga and a in gb):
                return True
        return False

    def _reach_reward(self, target_pos) -> float:
        return 1.0 - np.tanh(10.0 * float(np.linalg.norm(self._eef_pos() - target_pos)))

    def _obs(self) -> np.ndarray:
        qpos = self.d.qpos[self._arm_qpos]
        parts = [
            np.cos(qpos), np.sin(qpos), self.d.qvel[self._arm_qvel], self._eef_pos(),
            _mat2quat(self.d.site_xmat[self._eef_sid].reshape(3, 3)), self.d.qpos[self._grip_qpos],
        ]
        parts += self._object_obs()
        return np.concatenate(parts).astype(np.float32)

    def close(self):
        self.handler.close()


# --------------------------------------------------------------------- tasks
class NativeLiftEnv(NativeRobosuiteEnv):
    ENV_NAME = "Lift"
    PLACE_OBJECTS = [("cube_main", "cube_joint0", 0.831)]

    def _setup_task(self):
        self._cube = self._bid("cube_main")

    def _check_success(self) -> bool:
        return bool(self._body_pos(self._cube)[2] > _TABLE_TOP_Z + 0.04)

    def _task_reward(self) -> float:
        if self._check_success():
            return 2.25
        r = self._reach_reward(self._body_pos(self._cube))
        if self._grasping(self._geoms_of_body(self._cube)):
            r += 0.25
        return r

    def _object_obs(self):
        cube = self._body_pos(self._cube)
        return [cube, self.d.xquat[self._cube].copy(), cube - self._eef_pos()]


class NativeStackEnv(NativeRobosuiteEnv):
    ENV_NAME = "Stack"
    PLACE_OBJECTS = [("cubeA_main", "cubeA_joint0", 0.83), ("cubeB_main", "cubeB_joint0", 0.83)]
    XY_RANGE = 0.08

    def _setup_task(self):
        self._A, self._B = self._bid("cubeA_main"), self._bid("cubeB_main")
        self._gA, self._gB = self._geoms_of_body(self._A), self._geoms_of_body(self._B)

    def _check_success(self) -> bool:
        A = self._body_pos(self._A)
        lifted = A[2] > _TABLE_TOP_Z + 0.04
        touching = self._contact(self._gA, self._gB)
        return bool(lifted and touching and not self._grasping(self._gA))

    def _object_obs(self):
        return [self._body_pos(self._A), self.d.xquat[self._A].copy(), self._body_pos(self._B), self.d.xquat[self._B].copy()]


class NativeDoorEnv(NativeRobosuiteEnv):
    ENV_NAME = "Door"
    PLACE_OBJECTS = []  # door is welded; vendored placement is fixed

    def _setup_task(self):
        self._hinge_adr = self.m.jnt_qposadr[self._jid("Door_hinge")]
        self._handle = self._sid("door_handle") if self._sid("door_handle") >= 0 else None

    def _check_success(self) -> bool:
        return bool(self.d.qpos[self._hinge_adr] > 0.3)

    def _task_reward(self) -> float:
        return float(np.clip(self.d.qpos[self._hinge_adr] / 0.3, 0, 1)) + (1.0 if self._check_success() else 0.0)

    def _object_obs(self):
        return [np.array([self.d.qpos[self._hinge_adr]])]


class NativePickPlaceEnv(NativeRobosuiteEnv):
    """PickPlace base — success = target object in its bin (robosuite not_in_bin)."""

    ENV_NAME = "PickPlace"
    BUNDLE = "PickPlace"  # all variants share the 4-object model
    TARGET = "Can"  # Milk/Bread/Cereal/Can
    BIN_ID = 3  # milk:0 bread:1 cereal:2 can:3
    horizon = 500
    _BIN2 = np.array([0.1, 0.28, 0.8])
    _BIN_SIZE = np.array([0.39, 0.49, 0.82])

    def __init__(self, bundle_dir=None, seed=0):
        super().__init__(bundle_dir or (_BUNDLE_ROOT / self.BUNDLE.lower()), seed)

    def _setup_task(self):
        self._obj = self._bid(f"{self.TARGET}_main")

    def _in_bin(self, pos) -> bool:
        bx, by = self._BIN2[0], self._BIN2[1]
        if self.BIN_ID in (0, 2):
            bx -= self._BIN_SIZE[0] / 2
        if self.BIN_ID < 2:
            by -= self._BIN_SIZE[1] / 2
        return bool(
            bx < pos[0] < bx + self._BIN_SIZE[0] / 2
            and by < pos[1] < by + self._BIN_SIZE[1] / 2
            and self._BIN2[2] < pos[2] < self._BIN2[2] + 0.1
        )

    def _check_success(self) -> bool:
        return self._in_bin(self._body_pos(self._obj))

    def _object_obs(self):
        o = self._body_pos(self._obj)
        return [o, self.d.xquat[self._obj].copy(), o - self._eef_pos()]


class NativePickPlaceCanEnv(NativePickPlaceEnv):
    ENV_NAME, TARGET, BIN_ID = "PickPlaceCan", "Can", 3


class NativePickPlaceMilkEnv(NativePickPlaceEnv):
    ENV_NAME, TARGET, BIN_ID = "PickPlaceMilk", "Milk", 0


class NativePickPlaceBreadEnv(NativePickPlaceEnv):
    ENV_NAME, TARGET, BIN_ID = "PickPlaceBread", "Bread", 1


class NativePickPlaceCerealEnv(NativePickPlaceEnv):
    ENV_NAME, TARGET, BIN_ID = "PickPlaceCereal", "Cereal", 2


class NativeNutAssemblyEnv(NativeRobosuiteEnv):
    """NutAssembly base — success = target nut on its peg (robosuite on_peg + reach)."""

    ENV_NAME = "NutAssemblySquare"
    BUNDLE = "NutAssembly"  # square/round share the 2-nut model
    NUT = "SquareNut"
    PEG = "peg1"  # square->peg1, round->peg2
    horizon = 500

    def __init__(self, bundle_dir=None, seed=0):
        super().__init__(bundle_dir or (_BUNDLE_ROOT / self.BUNDLE.lower()), seed)

    def _setup_task(self):
        self._nut = self._bid(f"{self.NUT}_main")
        self._peg = self._bid(self.PEG)

    def _on_peg(self, obj_pos) -> bool:
        peg = self._body_pos(self._peg)
        return bool(
            abs(obj_pos[0] - peg[0]) < 0.03
            and abs(obj_pos[1] - peg[1]) < 0.03
            and obj_pos[2] < _TABLE_TOP_Z + 0.05
        )

    def _check_success(self) -> bool:
        nut = self._body_pos(self._nut)
        r_reach = 1 - np.tanh(10.0 * float(np.linalg.norm(self._eef_pos() - nut)))
        return bool(self._on_peg(nut) and r_reach < 0.6)

    def _object_obs(self):
        nut = self._body_pos(self._nut)
        return [nut, self.d.xquat[self._nut].copy(), nut - self._eef_pos()]


class NativeNutAssemblySquareEnv(NativeNutAssemblyEnv):
    ENV_NAME, NUT, PEG = "NutAssemblySquare", "SquareNut", "peg1"


class NativeNutAssemblyRoundEnv(NativeNutAssemblyEnv):
    ENV_NAME, NUT, PEG = "NutAssemblyRound", "RoundNut", "peg2"


NATIVE_ENVS = {
    "Lift": NativeLiftEnv,
    "Stack": NativeStackEnv,
    "Door": NativeDoorEnv,
    "PickPlaceCan": NativePickPlaceCanEnv,
    "PickPlaceMilk": NativePickPlaceMilkEnv,
    "PickPlaceBread": NativePickPlaceBreadEnv,
    "PickPlaceCereal": NativePickPlaceCerealEnv,
    "NutAssemblySquare": NativeNutAssemblySquareEnv,
    "NutAssemblyRound": NativeNutAssemblyRoundEnv,
}
