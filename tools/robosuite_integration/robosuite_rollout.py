"""Drive a native robosuite environment and capture reference data.

Two capture modes:

* :func:`robosuite_engine_rollout` — controller-free: reset the env, then step
  the *raw* MuJoCo actuators with a deterministic ctrl signal. This is the
  ground-truth physics trajectory the MetaSim loader must match bit-for-bit.

* :func:`robosuite_episode` — full closed-loop: step ``env.step(action)`` through
  robosuite's OSC controller, capturing flattened states, high-level actions,
  rewards, success flags and (optionally) rendered frames. This produces the
  demonstrations we replay in the MetaSim port.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from .common import CtrlFn, RolloutResult  # noqa: E402
from .inventory import RobosuiteTask, get  # noqa: E402


def make_robosuite_env(task: RobosuiteTask, *, render: bool = False, camera: str = "frontview"):
    """Build a native robosuite env for ``task`` with a fixed, deterministic config."""
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller=task.controller, robot=task.robot)
    env = robosuite.make(
        task.env_name,
        robots=task.robot,
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=False,
        render_camera=camera if render else None,
        camera_names=camera if render else None,
        camera_heights=480,
        camera_widths=480,
        control_freq=task.control_freq,
        horizon=task.horizon,
        ignore_done=True,
        hard_reset=True,
    )
    return env


def export_model_xml(task: RobosuiteTask, out_path: str | Path, *, seed: int = 0) -> dict:
    """Reset the env and dump its compiled MJCF (absolute asset paths) to disk.

    Returns a dict with the initial flattened state and model dims so callers can
    reproduce the exact starting condition in any engine.
    """
    env = make_robosuite_env(task)
    env.reset()
    np.random.seed(seed)
    env.reset()
    xml = env.model.get_xml()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml)

    m, d = env.sim.model._model, env.sim.data._data
    info = {
        "xml_path": str(out_path),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "qpos0": d.qpos.copy(),
        "qvel0": d.qvel.copy(),
        "dt": float(m.opt.timestep),
        "decimation": int(round((1.0 / task.control_freq) / m.opt.timestep)),
        "joint_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"j{i}" for i in range(m.njnt)],
        "actuator_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}" for i in range(m.nu)],
    }
    env.close()
    return info


def robosuite_engine_rollout(
    task: RobosuiteTask | str,
    *,
    n_steps: int,
    ctrl_builder: Callable[[int], CtrlFn],
    seed: int = 0,
) -> tuple[RolloutResult, str, dict[str, np.ndarray]]:
    """Controller-free rollout on robosuite's own ``MjSim``.

    ``ctrl_builder`` maps the model's actuator count ``nu`` to a concrete
    :data:`~tools.robosuite_integration.common.CtrlFn`, so the matching MetaSim
    rollout can be built from the *same* builder.

    Returns ``(rollout, xml, model_overrides)``. ``model_overrides`` holds the
    placement fields robosuite mutates on the *compiled* model at reset
    (e.g. welded-object/door ``body_pos``/``body_quat``), which ``get_xml()``
    does NOT capture — the MetaSim side must re-apply them to reproduce the
    exact model. Free-joint objects place via ``qpos`` and need no override.
    """
    task = get(task) if isinstance(task, str) else task
    env = make_robosuite_env(task)
    env.reset()
    np.random.seed(seed)
    env.reset()
    xml = env.model.get_xml()

    m, d = env.sim.model._model, env.sim.data._data
    dec = int(round((1.0 / task.control_freq) / m.opt.timestep))
    nu = m.nu
    ctrl_fn = ctrl_builder(nu)

    mujoco.mj_forward(m, d)
    qpos_log = [d.qpos.copy()]
    qvel_log = [d.qvel.copy()]
    ctrl_log: list[np.ndarray] = []
    time_log = [float(d.time)]

    for t in range(n_steps):
        ctrl = np.asarray(ctrl_fn(t, d.qpos.copy(), d.qvel.copy()), dtype=np.float64).reshape(nu)
        d.ctrl[:] = ctrl
        ctrl_log.append(ctrl.copy())
        for _ in range(dec):
            mujoco.mj_step(m, d)
        qpos_log.append(d.qpos.copy())
        qvel_log.append(d.qvel.copy())
        time_log.append(float(d.time))

    joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"j{i}" for i in range(m.njnt)]
    actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}" for i in range(nu)]
    result = RolloutResult(
        backend="robosuite_native",
        task_id=task.metasim_name,
        dt=float(m.opt.timestep),
        decimation=dec,
        qpos=np.stack(qpos_log),
        qvel=np.stack(qvel_log),
        ctrl=np.stack(ctrl_log),
        time=np.array(time_log),
        joint_names=joint_names,
        actuator_names=actuator_names,
    )
    model_overrides = {"body_pos": m.body_pos.copy(), "body_quat": m.body_quat.copy()}
    env.close()
    return result, xml, model_overrides


@dataclass
class RobosuiteEpisode:
    """Full closed-loop episode captured through robosuite's controller."""

    task_id: str
    env_name: str
    model_xml: str  # per-episode compiled MJCF
    states: np.ndarray  # (T+1, 1+nq+nv) flattened MjSim states
    actions: np.ndarray  # (T, action_dim) high-level actions
    rewards: np.ndarray  # (T,)
    success: np.ndarray  # (T,) bool
    object_obs: dict[str, np.ndarray]  # selected obs keys, each (T+1, dim)
    eef_pos: np.ndarray  # (T+1, 3)
    frames: list[np.ndarray]  # rendered RGB frames (may be empty)
    control_freq: int
    dt: float

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            task_id=self.task_id,
            env_name=self.env_name,
            model_xml=self.model_xml,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            success=self.success,
            eef_pos=self.eef_pos,
            control_freq=self.control_freq,
            dt=self.dt,
            object_obs_keys=np.array(list(self.object_obs.keys()), dtype=object),
        )
        for k, v in self.object_obs.items():
            payload[f"obs__{k}"] = v
        np.savez_compressed(path, **payload)
        return path


def robosuite_episode(
    task: RobosuiteTask | str,
    *,
    n_steps: int,
    policy: Callable[[int, dict], np.ndarray] | None = None,
    seed: int = 0,
    render: bool = False,
    camera: str = "frontview",
) -> RobosuiteEpisode:
    """Roll a full episode through robosuite, capturing everything needed to replay it."""
    task = get(task) if isinstance(task, str) else task
    env = make_robosuite_env(task, render=render, camera=camera)
    np.random.seed(seed)
    obs = env.reset()
    action_dim = env.action_dim

    if policy is None:
        # deterministic scripted probe: gentle sinusoid so the replay is non-trivial
        def policy(t: int, obs: dict) -> np.ndarray:  # noqa: ANN001
            a = 0.3 * np.sin(2 * np.pi * t / 50 + np.arange(action_dim) * 0.5)
            return a

    m = env.sim.model._model

    def flat_state() -> np.ndarray:
        return env.sim.get_state().flatten()

    def grab_obs(o: dict) -> tuple[dict[str, float], np.ndarray]:
        sel = {k: np.asarray(o[k], dtype=np.float64) for k in task.object_obs_keys if k in o}
        eef = np.asarray(o.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)
        return sel, eef

    states = [flat_state()]
    sel0, eef0 = grab_obs(obs)
    obs_logs: dict[str, list[np.ndarray]] = {k: [v] for k, v in sel0.items()}
    eef_log = [eef0]
    actions, rewards, success = [], [], []
    frames: list[np.ndarray] = []
    if render:
        frames.append(env.sim.render(camera_name=camera, width=480, height=480)[::-1])

    for t in range(n_steps):
        a = np.asarray(policy(t, obs), dtype=np.float64).reshape(action_dim)
        obs, r, done, info = env.step(a)
        actions.append(a.copy())
        rewards.append(float(r))
        success.append(bool(env._check_success()))
        states.append(flat_state())
        sel, eef = grab_obs(obs)
        for k, v in sel.items():
            obs_logs[k].append(v)
        eef_log.append(eef)
        if render:
            frames.append(env.sim.render(camera_name=camera, width=480, height=480)[::-1])

    episode = RobosuiteEpisode(
        task_id=task.metasim_name,
        env_name=task.env_name,
        model_xml=env.model.get_xml(),
        states=np.stack(states),
        actions=np.stack(actions),
        rewards=np.array(rewards),
        success=np.array(success, dtype=bool),
        object_obs={k: np.stack(v) for k, v in obs_logs.items()},
        eef_pos=np.stack(eef_log),
        frames=frames,
        control_freq=task.control_freq,
        dt=float(m.opt.timestep),
    )
    env.close()
    return episode
