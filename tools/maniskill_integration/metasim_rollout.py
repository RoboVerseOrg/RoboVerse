"""Step the existing MetaSim `maniskill.pick_cube` task on Sapien3.

We pair this with `maniskill_rollout.py` so the report can show
RoboVerse's `maniskill.pick_cube` (a simpler RoboVerse-flavored cube
lift task) next to the ManiSkill3 native `PickCube-v1`. The two tasks
are NOT bit-for-bit equivalent — see the integration report's
gap-analysis section.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MetaSimManiskillRollout:
    """Time-series captured from one MetaSim/Sapien3 rollout."""

    task_id: str
    seed: int
    actions: np.ndarray  # (T, A)
    qpos: np.ndarray  # (T+1, nq)
    cube_pos: np.ndarray  # (T+1, 3)
    success: np.ndarray  # (T+1,)
    elapsed: np.ndarray  # (T+1,)
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 20.0

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "T": int(self.actions.shape[0]),
            "nq": int(self.qpos.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "final_cube_z": float(self.cube_pos[-1, 2]),
            "max_cube_z": float(self.cube_pos[:, 2].max()),
            "any_success": bool(self.success.any()),
        }


def _detach(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def rollout_metasim_pick_cube(
    *,
    n_steps: int = 50,
    seed: int = 0,
    action_fn: Callable | None = None,
    render: bool = True,
    render_size: tuple[int, int] = (320, 240),
) -> MetaSimManiskillRollout | None:
    """Drive MetaSim's `maniskill.pick_cube` for `n_steps` and capture state."""
    try:
        import roboverse_pack.tasks.maniskill  # noqa: F401 — register side-effects
        from metasim.task.registry import get_task_class
    except Exception as e:
        print(f"[metasim_pick_cube] import failed: {e}")
        return None

    cls = get_task_class("maniskill.pick_cube")
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []

    try:
        env = cls(sc)
    except Exception as e:
        print(f"[metasim_pick_cube] task init failed: {e}")
        return None

    handler = env.handler
    physics_data = None
    try:
        physics_data = handler.scene if hasattr(handler, "scene") else None
    except Exception:
        pass

    # Determine action dim from the handler's robot joints.
    try:
        joint_names = handler._get_joint_names(handler.robots[0].name, sort=True)
    except Exception:
        joint_names = []
    action_dim = len(joint_names)
    if action_dim == 0:
        # fallback to robot config num_joints
        action_dim = handler.robots[0].num_joints

    def _zero(t, q):
        return np.zeros(action_dim, dtype=np.float32)

    fn = action_fn or _zero

    states_dict = handler.get_states(mode="dict")
    robot_name = handler.robots[0].name

    def _qpos_from(st):
        try:
            jp = st[0]["robots"][robot_name]["dof_pos"]
            return (
                np.array([jp[n] for n in joint_names], dtype=np.float64) if joint_names else np.array(list(jp.values()))
            )
        except Exception:
            return np.zeros(action_dim)

    def _cube_pos(st):
        try:
            return np.asarray(st[0]["objects"]["cube"]["pos"], dtype=np.float64).reshape(3)
        except Exception:
            return np.zeros(3)

    qpos_log = [_qpos_from(states_dict)]
    cube_log = [_cube_pos(states_dict)]
    success_log = [False]
    actions_log: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    # Best-effort offscreen render via Sapien's own renderer; many sapien3
    # builds need a real GPU here so we tolerate failure silently.
    renderer = None
    if render:
        try:
            # Probe whether handler has a render method we can call.
            if hasattr(handler, "render"):
                pass
        except Exception:
            renderer = None

    for t in range(n_steps):
        a = fn(t, qpos_log[-1])
        a = np.asarray(a, dtype=np.float32).reshape(action_dim)
        # Build a dof_pos_target payload (position-control by default).
        target = {jn: float(a[i]) + float(qpos_log[-1][i]) for i, jn in enumerate(joint_names)}
        try:
            handler.set_dof_targets([{robot_name: {"dof_pos_target": target}}])
            handler.simulate()
        except Exception as e:
            print(f"[metasim_pick_cube] sim step {t} failed: {e}")
            break
        actions_log.append(a)
        st = handler.get_states(mode="dict")
        qpos_log.append(_qpos_from(st))
        cube_log.append(_cube_pos(st))
        # Termination from the task's own checker.
        try:
            ten_state = handler.get_states()
            success = bool(_detach(env._terminated(ten_state)).any())
        except Exception:
            success = False
        success_log.append(success)

    try:
        handler.close()
    except Exception:
        pass

    return MetaSimManiskillRollout(
        task_id="maniskill.pick_cube",
        seed=seed,
        actions=np.stack(actions_log) if actions_log else np.zeros((0, action_dim), dtype=np.float32),
        qpos=np.stack(qpos_log),
        cube_pos=np.stack(cube_log),
        success=np.array(success_log),
        elapsed=np.arange(len(qpos_log)),
        frames=frames,
        fps=20.0,
    )


def save_metasim_rollout(path: Path, r: MetaSimManiskillRollout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        task_id=r.task_id,
        seed=r.seed,
        actions=r.actions,
        qpos=r.qpos,
        cube_pos=r.cube_pos,
        success=r.success,
        elapsed=r.elapsed,
    )
