"""NativeCokeCanEnv — full reset()/step()/get_obs(), pure SAPIEN, zero upstream dep.

Assembles the vendored pieces (scene + combined controller + grasp checker + success
evaluator + overlay) into a complete episode loop that reproduces
``GraspSingleOpenedCokeCanInScene-v0`` on RoboVerse/MetaSim's own SAPIEN 2.x with NO
``mani_skill2_real2sim`` / ``simpler_env`` import — the deletable native port.

Reset RNG is reproduced from the upstream ManiSkill2 flow (verified against upstream):
  * orientation = ``RandomState(seed).choice([upright, laid_vertically, lr_switch])``
  * a fresh ``RandomState(seed)`` then draws obj_init_xy, robot init_x, init_y (in that order),
    because BaseEnv re-seeds the episode RNG immediately before ``initialize_episode``.
``random_choice`` on the 1-element model_ids / scales lists does not consume the RNG.

Step drives the real 7-D action through the vendored CombinedController (171 substeps), then
computes ``info["success"]`` via the vendored grasp checker + PickSuccessEvaluator. Observation
is the overhead camera RGB (with greenscreen overlay) + agent qpos — what the policy consumes.
"""

from __future__ import annotations

import numpy as np
import sapien.core as sapien

from .control import CombinedController
from .grasp import GoogleRobotGraspChecker, PickSuccessEvaluator
from .robot_config import google_robot_deployed_controller_configs
from .scene import CONTROL_FREQ, INIT_QPOS, ROBOT_INIT_HEIGHT, SIM_FREQ, NativeCokeCanScene


def _orientation_quats():
    from transforms3d.euler import euler2quat

    return {
        "upright": euler2quat(np.pi / 2, 0, 0),
        "laid_vertically": euler2quat(0, 0, np.pi / 2),
        "lr_switch": euler2quat(0, 0, np.pi),
    }


SCENE_TABLE_HEIGHT = 0.87
MAX_EPISODE_STEPS = 80
SUBSTEPS = SIM_FREQ // CONTROL_FREQ


class NativeCokeCanEnv:
    def __init__(self, *, default_orientation=None, instruction="pick coke can", **scene_paths):
        self.scene_obj = NativeCokeCanScene(**scene_paths)
        self.controller = None
        self.grasp_checker = None
        self.evaluator = PickSuccessEvaluator()
        self._built = False
        self._elapsed = 0
        self._orient_quats = _orientation_quats()
        self.obj_height_after_settle = None
        # default object orientation (matches upstream variant ctor flags:
        # lr_switch/laid_vertically/upright); None => drawn from the episode seed.
        self.default_orientation = default_orientation
        self._instruction = instruction

    # ---- build ---------------------------------------------------------------
    def build(self):
        self.scene_obj.build()
        cfgs = google_robot_deployed_controller_configs()
        self.controller = CombinedController(cfgs, self.scene_obj.robot, CONTROL_FREQ, sim_freq=SIM_FREQ)
        self.grasp_checker = GoogleRobotGraspChecker(self.scene_obj.robot, self.scene_obj.scene)
        self._built = True
        return self

    # ---- reset ---------------------------------------------------------------
    def reset(self, seed=0, options=None):
        if not self._built:
            self.build()
        options = options or {}
        robot = self.scene_obj.robot
        obj = self.scene_obj.obj

        # --- reproduce upstream reset RNG (draw order verified vs upstream) ---
        obj_init_options = (options.get("obj_init_options") or {}).copy()
        if obj_init_options.get("init_rot_quat") is None:
            orientation = obj_init_options.get("orientation", self.default_orientation)
            if orientation is None:
                orientation = np.random.RandomState(seed).choice(list(self._orient_quats.keys()))
            obj_init_rot_quat = self._orient_quats[orientation]
        else:
            obj_init_rot_quat = np.array(obj_init_options["init_rot_quat"])
            orientation = obj_init_options.get("orientation")

        rng = np.random.RandomState(seed)  # fresh, re-seeded as BaseEnv does
        obj_init_xy = obj_init_options.get("init_xy")
        if obj_init_xy is None:
            obj_init_xy = rng.uniform([-0.35, -0.02], [-0.12, 0.42], [2])
        obj_init_z = obj_init_options.get("init_z", SCENE_TABLE_HEIGHT) + 0.5

        robot_init_options = options.get("robot_init_options") or {}
        robot_init_xy = robot_init_options.get("init_xy")
        if robot_init_xy is None:
            init_x = rng.uniform(0.30, 0.40)
            init_y = rng.uniform(0.0, 0.2)
            robot_init_xy = [init_x, init_y]
        robot_init_quat = robot_init_options.get("init_rot_quat", [0, 0, 0, 1])

        # --- _initialize_actors: object falls + settles (robot moved away) ---
        robot.set_root_pose(sapien.Pose([-10, 0, 0]))
        robot.set_qpos(INIT_QPOS)
        robot.set_qvel(np.zeros(robot.dof))
        obj.set_pose(sapien.Pose(np.hstack([obj_init_xy, obj_init_z]), obj_init_rot_quat))
        obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)
        obj.lock_motion(0, 0, 0, 0, 0, 0)
        obj.set_pose(obj.pose)
        obj.set_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)
        if np.linalg.norm(obj.velocity) > 1e-3 or np.linalg.norm(obj.angular_velocity) > 1e-2:
            self._settle(1.5)
        self.obj_height_after_settle = float(obj.pose.p[2])

        # --- _initialize_agent: place robot at its init pose, hold qpos ---
        # (upstream sets qpos directly; drive targets are set on the first step's set_action,
        #  and no physics steps between here and the first action, so qpos holds.)
        robot.set_root_pose(sapien.Pose([robot_init_xy[0], robot_init_xy[1], ROBOT_INIT_HEIGHT], robot_init_quat))
        robot.set_qpos(INIT_QPOS)
        robot.set_qvel(np.zeros(robot.dof))
        self.controller.reset()

        self._elapsed = 0
        self.evaluator.reset(self.obj_height_after_settle)
        return self.get_obs(), {"orientation": orientation}

    def _settle(self, t):
        scene = self.scene_obj.scene
        for _ in range(int(SIM_FREQ * t)):
            scene.step()

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        scene = self.scene_obj.scene
        self.controller.set_action(np.asarray(action, dtype=np.float32))
        for _ in range(SUBSTEPS):
            self.controller.before_simulation_step()
            scene.step()
        self._elapsed += 1
        obs = self.get_obs()
        info = self._evaluate()
        reward = 1.0 if info["success"] else 0.0
        terminated = bool(info["success"])
        truncated = self._elapsed >= MAX_EPISODE_STEPS
        return obs, reward, terminated, truncated, info

    # ---- obs + eval ----------------------------------------------------------
    def get_obs(self):
        rgb = self.scene_obj.render_obs()
        return {
            "image": {"overhead_camera": {"rgb": rgb}},
            "agent": {"qpos": self.scene_obj.robot.get_qpos().copy()},
            "extra": {},
        }

    def get_language_instruction(self):
        return self._instruction

    def _evaluate(self):
        obj = self.scene_obj.obj
        robot = self.scene_obj.robot
        is_grasped = self.grasp_checker.check_grasp(obj, max_angle=80)
        contacts = self.scene_obj.scene.get_contacts()
        robot_link_names = [lk.get_name() for lk in robot.get_links()]
        return self.evaluator.step(
            is_grasped=is_grasped,
            obj_z=float(obj.pose.p[2]),
            contacts=contacts,
            obj=obj,
            robot_link_names=robot_link_names,
        )
