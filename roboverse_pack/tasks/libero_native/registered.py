"""Register the 130 native LIBERO tasks so they are discoverable via the task registry.

``get_task_class("libero_native/<suite>/<task>")`` returns a :class:`BaseTaskEnv`
that runs the task entirely on MuJoCo + NumPy — **no ``libero`` / ``robosuite``
import**. Unlike the standard handler-backed tasks, this wrapper drives the
self-contained :class:`~.native_task.LiberoNativeTask` (vendored MJCF + shared
assets resolved from ``roboverse_data`` / HF), so it is the library-free path.

Registration is **cheap**: the task list is shipped (``_tasks.json``) and only the
class objects are created at import — the scene MJCF + assets are loaded lazily on
first instantiation (clone or HF download), matching the existing LIBERO packs.

    from metasim.task.registry import get_task_class
    Task = get_task_class("libero_native/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket")
    env = Task()                       # loads the vendored scene; no libero
    obs = env.reset()                  # deterministic BDDL reset state
    obs, reward, success, timeout, _ = env.step([0, 0, -0.1, 0, 0, 0, -1])  # OSC delta + gripper
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task

_TASKS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tasks.json")))


class LiberoNativeRegisteredTask(BaseTaskEnv):
    """Discoverable wrapper around a self-contained native LIBERO task (no libero).

    Drives :class:`~.native_task.LiberoNativeTask`: the scene, ported BDDL success
    checker, inlined OSC_POSE controller and renderer all run on MuJoCo + NumPy.
    Single-env, CPU. The arm follows the bitwise OSC port; the gripper command
    (action[6] in ``[-1, 1]``) is linearly mapped onto the gripper actuator's
    ``ctrlrange``.
    """

    suite: str = ""
    task: str = ""
    max_episode_steps = 300
    scenario = None  # intentionally not the standard handler-backed scenario path

    def __init__(self, scenario=None, device: str | torch.device | None = None) -> None:
        from .native_task import LiberoNativeTask

        self._t = LiberoNativeTask.from_vendored(self.suite, self.task)
        if self._t.init_state is None:
            raise RuntimeError(
                f"vendored bundle for {self.suite}/{self.task} has no init_state; re-run migrate_libero_assets"
            )
        self.num_envs = 1
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.scenario = scenario
        self.handler = None
        self._episode_steps = torch.zeros(1, dtype=torch.int32, device=self.device)
        # gripper action -> ctrl: midpoint ± half-range of the gripper actuators
        gr = self._t.model.actuator_ctrlrange[self._t.grip_act]
        self._grip_mid = float(gr[:, 0].mean() + gr[:, 1].mean()) / 2.0
        self._grip_half = float(gr[:, 1].mean() - gr[:, 0].mean()) / 2.0
        self._prepare_callbacks()

    def _obs(self) -> dict:
        return {
            "joint_qpos": torch.tensor(self._t.arm_qpos(), dtype=torch.float32, device=self.device).unsqueeze(0),
            "success": torch.tensor([self._t.success()], dtype=torch.bool, device=self.device),
        }

    def reset(self, states=None, env_ids=None):
        """Reset to the bundled deterministic BDDL state (or a provided flat state)."""
        st = self._t.init_state if states is None else np.asarray(states)
        self._t.reset(st)
        self._episode_steps.zero_()
        return self._obs()

    def step(self, actions):
        """One OSC_POSE policy step: ``[Δx,Δy,Δz,Δrx,Δry,Δrz, grip]``."""
        a = np.asarray(actions, dtype=float).reshape(-1)
        grip_cmd = float(a[6]) if a.size > 6 else 0.0
        self._t.step(a[:6], self._grip_mid + self._grip_half * np.clip(grip_cmd, -1.0, 1.0))
        self._episode_steps += 1
        success = torch.tensor([self._t.success()], dtype=torch.bool, device=self.device)
        time_out = self._episode_steps >= self.max_episode_steps
        return self._obs(), success.float(), success, time_out, {}

    def success(self) -> bool:
        """Ported BDDL goal checker (bitwise vs ``env._check_success``)."""
        return bool(self._t.success())

    def render(self, *, height: int = 256, width: int = 256, camera: str = "agentview"):
        """Agentview RGB of the current state (matches robosuite geom groups)."""
        return self._t.render(camera=camera, height=height, width=width)

    def close(self) -> None:
        """Release the offscreen renderer, if one was created."""
        r = getattr(self._t, "_renderer", None)
        if r is not None:
            r.close()


def _make_task_class(suite: str, task: str) -> type:
    cls = type(
        f"LiberoNative_{suite}_{task}",
        (LiberoNativeRegisteredTask,),
        {"suite": suite, "task": task, "__doc__": f"Native LIBERO task {suite}/{task} (no libero/robosuite)."},
    )
    return register_task(f"libero_native/{suite}/{task}")(cls)


REGISTERED = [_make_task_class(t["suite"], t["task"]) for t in _TASKS]
