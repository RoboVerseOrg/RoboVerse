"""Shared base for the MetaSim-API SimplerEnv tasks.

Every SimplerEnv task is a ``BaseTaskEnv`` whose ``ScenarioCfg`` is loaded by the MetaSim sapien2
handler (robot / scene / objects / mounted intrinsic camera / SceneConfig). This base wires the
handler-loaded objects to the SimplerEnv-specific add-ons (the vendored EE controller, the
visual-matching overlay, the success checker) and runs the high-freq control loop on the handler's
articulation. Subclasses supply the scenario + the family-specific reset / evaluate.
"""

from __future__ import annotations

import os

import numpy as np

from metasim.scenario.objects import RigidObjCfg
from metasim.task.base import BaseTaskEnv

from .._native.overlay import apply_greenscreen_overlay, load_overlay_image

HIDDEN_POS = (0.0, 0.0, -100.0)  # park inactive candidate objects far below, out of frame/contact


def set_actor_collision(obj, enabled: bool):
    """Toggle an actor's collision (groups). Parked inactive candidate objects get collision OFF so
    they don't perturb the active object's contact solve (keeps parity bitwise); active ones ON."""
    g = (1, 1, 0, 0) if enabled else (0, 0, 0, 0)
    for cs in obj.get_collision_shapes():
        cs.set_collision_groups(*g)


def _visual_mesh(model_dir):
    for n in ("textured.obj", "textured.dae", "textured.glb"):
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(model_dir)


def candidate_mesh_objects(models_dir, model_ids, *, friction=None, damping=0.1, density_map=None):
    """Mesh RigidObjCfgs for a family's full candidate set (random-object tasks declare them all in
    the ScenarioCfg; reset activates the per-episode subset and parks the rest at ``HIDDEN_POS``)."""
    cfgs = []
    for mid in model_ids:
        md = os.path.join(models_dir, mid)
        cfgs.append(
            RigidObjCfg(
                name=mid,
                mesh_path=_visual_mesh(md),
                collision_mesh_path=os.path.join(md, "collision.obj"),
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0 if friction is not None else None,
                mesh_density=(density_map or {}).get(mid),
                linear_damping=damping,
                angular_damping=damping,
                fix_base_link=False,
                default_position=HIDDEN_POS,
            )
        )
    return cfgs


def apply_external_states(task, states, env_ids=None) -> bool:
    """Apply caller-supplied ``states`` on top of a SimplerEnv task's scripted reset.

    ``BaseTaskEnv.reset(states=...)`` means "put the selected envs in these states instead of the
    task's initial ones"; the IL/VLA eval runners replay demo initial states that way. A SimplerEnv
    reset scripts its own layout (per-episode RNG, fall-and-settle), so external states are applied
    once that layout is in place and before the controller / success checker latch onto it — the
    caller's states win, exactly as in the base contract.

    Args:
        task: The SimplerEnv task (any ``BaseTaskEnv``; used for its handler).
        states: External states, or None to keep the task's scripted initial states.
        env_ids: The environment ids the states apply to.

    Returns:
        True if states were applied, False if ``states`` was None. Callers use this to refresh any
        cached quantity they derived from the scripted layout (e.g. post-settle object heights).
    """
    if states is None:
        return False
    task.handler.set_states(states=states, env_ids=env_ids)
    task.handler.refresh_render()
    return True


class SimplerMetaSimTask(BaseTaskEnv):
    """Base task: handler loads the scenario; SimplerEnv control/overlay/checker layer on top."""

    scenario = None
    max_episode_steps = 80

    # subclass overrides
    robot_name = "robot"
    cam_name = "camera"
    sim_freq = 513
    control_freq = 3
    overlay_path = None
    instruction = ""

    def __init__(self, scenario=None, device=None, **kwargs):
        self._task_kwargs = kwargs
        super().__init__(scenario if scenario is not None else self._build_scenario(), device=device)
        self.substeps = self.sim_freq // self.control_freq
        self.scene = self.handler.scene
        self.robot = self.handler.object_ids[self.robot_name]
        self.camera = self.handler.camera_ids[self.cam_name]
        self._install_lighting()
        self.controller = self._make_controller()
        self._overlay = load_overlay_image(self.overlay_path) if self.overlay_path else None
        self._elapsed = 0
        self._wire()

    # ---- hooks subclasses must/may implement ---------------------------------
    def _build_scenario(self):
        raise NotImplementedError

    def _install_lighting(self):
        """Install SimplerEnv lighting on the handler scene (default lights were skipped)."""
        raise NotImplementedError

    def _make_controller(self):
        raise NotImplementedError

    def _wire(self):
        """Family-specific one-time wiring (friction materials, checkers, object handles)."""

    def _evaluate(self) -> dict:
        raise NotImplementedError

    def _foreground_ids(self):
        """Actor ids kept as foreground for the greenscreen overlay (robot + manipulated objects)."""
        return [link.get_id() for link in self.robot.get_links()]

    def get_language_instruction(self):
        return self.instruction

    # ---- shared machinery ----------------------------------------------------
    def _settle(self, t):
        for _ in range(int(self.sim_freq * t)):
            self.scene.step()

    def render_color(self):
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_float_texture("Color")[..., :3]

    def get_obs(self, overlay=True):
        color = self.render_color()
        if not overlay or self._overlay is None:
            rgb = np.clip(color * 255, 0, 255).astype(np.uint8)
        else:
            seg = self.camera.get_uint32_texture("Segmentation")[..., 1]
            ov = apply_greenscreen_overlay(
                color.astype(np.float64), seg, self._overlay, foreground_actor_ids=self._foreground_ids()
            )
            rgb = np.clip(ov * 255, 0, 255).astype(np.uint8)
        return {"image": {self.cam_name: {"rgb": rgb}}, "agent": {"qpos": self.robot.get_qpos().copy()}}

    def step(self, action):
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(self.substeps):
            self.controller.before_simulation_step()
            self.scene.step()
        self._elapsed += 1
        info = self._evaluate()
        return (
            self.get_obs(),
            1.0 if info["success"] else 0.0,
            bool(info["success"]),
            self._elapsed >= self.max_episode_steps,
            info,
        )
