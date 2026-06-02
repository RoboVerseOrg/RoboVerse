"""Run a LIBERO task entirely inside MetaSim's MuJoCo substrate — NO ``libero``.

Loads a self-contained bundle (``model.mjb`` + init state + demo actions + grip
ctrl + resolved BDDL goal/OSC config, produced once by
``scripts.native.export_libero_task``) and runs it with the inlined OSC_POSE
controller (:mod:`.osc`) + the ported BDDL checker (:mod:`.checker`). This module
imports only ``mujoco`` + ``numpy`` + this package — proving a LIBERO task's
scene, control law and success criterion all run without the upstream library.

    from roboverse_pack.tasks.libero_native.native_task import LiberoNativeTask
    task = LiberoNativeTask("alphabet_soup")
    print(task.replay_demo())   # {'first_success_step': ..., 'final_success': bool}
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from . import checker as nc
from .osc import NativeOSCPose

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class LiberoNativeTask:
    """A LIBERO task loaded from a self-contained bundle; runs with no libero."""

    def __init__(self, name: str, bundle_dir: str | None = None):
        d = bundle_dir or os.path.join(ASSETS, name)
        self.model = mujoco.MjModel.from_binary_path(os.path.join(d, "model.mjb"))
        self.data = mujoco.MjData(self.model)
        meta = json.load(open(os.path.join(d, "goal.json")))
        self.goal = meta["goal"]
        cfg = meta["osc"]
        self.substeps = cfg["substeps"]
        self.arm_act = np.asarray(cfg["arm_act_index"], int)
        self.grip_act = np.asarray(cfg["grip_act_index"], int)
        self.init_state = np.load(os.path.join(d, "init_states.npy"))
        self.demo_actions = np.load(os.path.join(d, "demo_actions.npy"))
        self.grip_ctrl = np.load(os.path.join(d, "grip_ctrl.npy"))
        self.osc = NativeOSCPose(
            self.model,
            self.data,
            cfg["eef_site"],
            cfg["arm_qvel_index"],
            cfg["arm_act_index"],
            initial_joint=cfg["initial_joint"],
            torque_limits=(np.asarray(cfg["torque_limits"][0]), np.asarray(cfg["torque_limits"][1])),
        )

    def render(self, camera: str = "agentview", height: int = 256, width: int = 256):
        """Offscreen RGB render of the current state from the bundled MuJoCo model.

        Returns a right-side-up ``(H, W, 3)`` uint8 image (the model's camera is the
        OpenGL bottom-up convention; ``mujoco.Renderer`` already yields it top-down).
        The geom-group mask matches robosuite's offscreen ``vopt.geomgroup``
        (``[0,1,1,0,0,0]`` — show visual groups 1/2, hide collision group 0), so the
        render is **bitwise-identical** to LIBERO's ``agentview_image`` at the same
        state. Library-free — proves the native task is also renderable without libero.
        """
        if getattr(self, "_renderer", None) is None or self._renderer.height != height or self._renderer.width != width:
            if getattr(self, "_renderer", None) is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._vopt = mujoco.MjvOption()
            for g in (0, 3, 4, 5):  # robosuite shows only geom groups 1 and 2
                self._vopt.geomgroup[g] = 0
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._vopt)
        return self._renderer.render()

    def reset(self, state=None):
        """Set the model to a flat ``[time, qpos, qvel]`` state (default: bundled init)."""
        st = self.init_state if state is None else state
        nq, nv = self.model.nq, self.model.nv
        self.data.qpos[:] = st[1 : 1 + nq]
        self.data.qvel[:] = st[1 + nq : 1 + nq + nv]
        mujoco.mj_forward(self.model, self.data)

    def step(self, action, grip_ctrl):
        """One policy step: OSC sets the goal, then run + step for `substeps`."""
        self.osc.set_goal(np.asarray(action)[:6])
        self.data.ctrl[self.grip_act] = grip_ctrl
        for _ in range(self.substeps):
            self.data.ctrl[self.arm_act] = self.osc.run()
            self.data.ctrl[self.grip_act] = grip_ctrl
            mujoco.mj_step(self.model, self.data)

    def success(self) -> bool:
        """Evaluate the ported BDDL goal against the current state."""
        return nc.check_success(self.model, self.data, self.goal)

    def arm_qpos(self):
        """Current arm joint positions."""
        return np.array(self.data.qpos[self.osc.qvi])

    def replay_demo(self):
        """Replay the bundled demo; return per-step success summary."""
        self.reset()
        first, final = None, False
        for t in range(len(self.demo_actions)):
            self.step(self.demo_actions[t], self.grip_ctrl[t])
            s = self.success()
            if s and first is None:
                first = t
            final = s
        return {"steps": len(self.demo_actions), "first_success_step": first, "final_success": final}
