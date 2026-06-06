"""Per-task scene specs for the MetaSim-native ManiSkill tabletop tasks.

Captured one-to-one from native ManiSkill (seed 0) — robot mount pose and every actor's primitive
geometry / mass / colour / spawn. Used by :mod:`native_tasks` to author the shipped tasks without a
runtime ``mani_skill`` import. Only single-primitive tasks live here; multi-shape actors
(PullCubeTool L-tool, PegInsertionSide box-with-hole, PlugCharger) need a multi-box object cfg and
are tracked separately.

The ``success`` callbacks are simple, honest geometric proxies (they read the live object poses);
porting ManiSkill's exact goal-site + robot-static success and the dense rewards is a follow-up.
"""

from __future__ import annotations

from . import rewards as RW
from . import success as SU

# Default panda mount (table edge); RollBall rotates it.
_BASE = ((-0.615, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
_CUBE = [0.04, 0.04, 0.04]
_RED, _GREEN, _BLUE, _MSBLUE = [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.047, 0.165, 0.627]


def _lifted(name, z=0.12):
    return lambda p: p[name][2] > z


def _stacked(top, base, tol=0.01):
    def f(p):
        a, b = p[top], p[base]
        xy = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        return xy < 0.02 + tol and abs(a[2] - (b[2] + 0.04)) < tol
    return f


def _moved(name, spawn, dist=0.05):
    sx, sy = spawn[0], spawn[1]
    return lambda p: ((p[name][0] - sx) ** 2 + (p[name][1] - sy) ** 2) ** 0.5 > dist


# --- fully-ported (native 1:1) reset / reward / success for the wired tasks ---
def _set_cube(task, name, x, y, z):
    import sapien

    task.handler.object_ids[name].set_pose(sapien.Pose([float(x), float(y), float(z)], [1.0, 0.0, 0.0, 0.0]))


def _pick_cube_reset(task, rng):
    """ManiSkill PickCube reset: cube XY ∈ U(±0.1); goal XY ∈ U(±0.1), Z = U(0,0.3)+0.02."""
    cx, cy = rng.uniform(-0.1, 0.1, 2)
    _set_cube(task, "cube", cx, cy, 0.02)
    gx, gy = rng.uniform(-0.1, 0.1, 2)
    return [gx, gy, float(rng.uniform(0, 0.3)) + 0.02]


def _pick_cube_reward(task):
    return RW.pick_cube(cube_pos=task.obj_pos("cube"), tcp_pos=task.tcp_pos(), goal_pos=task.goal_pos,
                        qvel_arm=task.qvel_arm(), is_grasped=task.is_grasped("cube"),
                        is_robot_static=task.is_robot_static())


def _pick_cube_success(task):
    return SU.pick_cube(cube_pos=task.obj_pos("cube"), goal_pos=task.goal_pos,
                        is_robot_static=task.is_robot_static())


def _push_cube_reset(task, rng):
    """ManiSkill PushCube reset: cube XY ∈ U(±0.1); goal = cube + [0.2, 0, 0] (goal_radius 0.1)."""
    cx, cy = rng.uniform(-0.1, 0.1, 2)
    _set_cube(task, "cube", cx, cy, 0.02)
    return [cx + 0.2, cy, 1e-3]


def _push_cube_reward(task):
    return RW.push_cube(cube_pos=task.obj_pos("cube"), tcp_pos=task.tcp_pos(), goal_pos=task.goal_pos)


def _push_cube_success(task):
    return SU.push_cube(cube_pos=task.obj_pos("cube"), goal_pos=task.goal_pos)


# name -> spec. ``objects``: list of (name, kind, geom, mass, color, pos, kinematic).
#   box geom = full size [x,y,z]; sphere geom = radius.
TASK_SPECS: dict[str, dict] = {
    "pick_cube": {
        "gym_id": "PickCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [("cube", "box", _CUBE, 0.064, _RED, (-0.0007, 0.0536, 0.02), False)],
        "success": _lifted("cube"),  # fallback proxy (unused — success_full set below)
        "goal": _pick_cube_reset, "reward": _pick_cube_reward, "success_full": _pick_cube_success,
    },
    "push_cube": {
        "gym_id": "PushCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [("cube", "box", _CUBE, 0.064, _MSBLUE, (-0.0007, 0.0536, 0.02), False)],
        "success": _moved("cube", (-0.0007, 0.0536)),
        "goal": _push_cube_reset, "reward": _push_cube_reward, "success_full": _push_cube_success,
    },
    "pull_cube": {
        "gym_id": "PullCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [("cube", "box", _CUBE, 0.064, _MSBLUE, (-0.0007, 0.0536, 0.02), False)],
        "success": _moved("cube", (-0.0007, 0.0536)),
    },
    "stack_cube": {
        "gym_id": "StackCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cubeA", "box", _CUBE, 0.064, _RED, (-0.0831, -0.0935, 0.02), False),
            ("cubeB", "box", _CUBE, 0.064, _GREEN, (-0.0393, 0.1073, 0.02), False),
        ],
        "success": _stacked("cubeA", "cubeB"),
    },
    "poke_cube": {
        "gym_id": "PokeCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cube", "box", _CUBE, 0.064, _RED, (0.2193, -0.0385, 0.02), False),
            ("peg", "box", [0.24, 0.05, 0.05], 0.6, _MSBLUE, (-0.0007, 0.0536, 0.025), False),
        ],
        "success": _moved("cube", (0.2193, -0.0385)),
    },
    "lift_peg_upright": {
        "gym_id": "LiftPegUpright-v1", "base": _BASE, "max_steps": 50,
        "objects": [("peg", "box", [0.24, 0.05, 0.05], 0.6, [0.69, 0.055, 0.055], (-0.0007, 0.0536, 0.025), False)],
        "success": _lifted("peg", 0.1),
    },
    "roll_ball": {
        "gym_id": "RollBall-v1", "base": ((-0.10, 1.0, 0.0), (0.7071068, 0.0, 0.0, -0.7071068)),
        "max_steps": 80,
        "objects": [("ball", "sphere", 0.035, 0.17959, _BLUE, (-0.1022, 0.6536, 0.035), False)],
        "success": _moved("ball", (-0.1022, 0.6536), 0.1),
    },
    "place_sphere": {
        "gym_id": "PlaceSphere-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("sphere", "sphere", 0.02, 0.03351, _MSBLUE, (-0.0752, 0.0536, 0.02), False),
            ("bin", "box", [0.05, 0.05, 0.005], 0.0225, [1, 1, 1], (0.0088, -0.0736, 0.0025), True),
        ],
        "success": _moved("sphere", (-0.0752, 0.0536)),
    },
    "stack_pyramid": {
        "gym_id": "StackPyramid-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cubeA", "box", _CUBE, 0.064, _RED, (-0.0007, 0.1073, 0.02), False),
            ("cubeB", "box", _CUBE, 0.064, _GREEN, (-0.0823, -0.1472, 0.02), False),
            ("cubeC", "box", _CUBE, 0.064, _BLUE, (-0.0385, 0.0536, 0.02), False),
        ],
        "success": _stacked("cubeC", "cubeA"),
    },
    # --- multi-shape (compound-box) tasks, via PrimitiveMultiBoxCfg ---
    "pull_cube_tool": {
        "gym_id": "PullCubeTool-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cube", "box", _CUBE, 0.064, _MSBLUE, (0.0677, -0.2104, 0.025), False),
            ("l_shape_tool", "multibox", [
                {"half_size": [0.10, 0.025, 0.025], "pos": [0.10, 0.0, 0.0]},
                {"half_size": [0.025, 0.05, 0.025], "pos": [0.175, 0.05, 0.0]},
            ], 0.5, _RED, (-0.1993, -0.2536, 0.025), False),
        ],
        "success": _moved("cube", (0.0677, -0.2104)),
    },
    "peg_insertion_side": {
        "gym_id": "PegInsertionSide-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("peg_0", "box", [0.214, 0.0444, 0.0444], 0.41986, [0.843, 0.173, 0.094], (-0.0007, -0.0695, 0.0222), False),
            ("box_with_hole_0", "multibox", [
                {"half_size": [0.107, 0.0365, 0.107], "pos": [0.0, 0.0704, 0.0]},
                {"half_size": [0.107, 0.0453, 0.107], "pos": [0.0, -0.0617, 0.0]},
                {"half_size": [0.107, 0.107, 0.039], "pos": [0.0, 0.0, 0.068]},
                {"half_size": [0.107, 0.107, 0.0428], "pos": [0.0, 0.0, -0.0641]},
            ], 14.97128, [1.0, 0.652, 0.255], (0.0134, 0.298, 0.107), True),
        ],
        "success": _moved("peg_0", (-0.0007, -0.0695)),
    },
    "plug_charger": {
        "gym_id": "PlugCharger-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("charger", "multibox", [
                {"half_size": [0.008, 0.0008, 0.0032], "pos": [0.008, 0.007, 0.0]},
                {"half_size": [0.008, 0.0008, 0.0032], "pos": [0.008, -0.007, 0.0]},
                {"half_size": [0.02, 0.015, 0.012], "pos": [-0.02, 0.0, 0.0]},
            ], 0.02911, [1, 1, 1], (-0.0496, 0.1661, 0.012), False),
            ("receptacle", "multibox", [
                {"half_size": [0.01, 0.05, 0.0232], "pos": [-0.01, 0.0, 0.0268]},
                {"half_size": [0.01, 0.05, 0.0232], "pos": [-0.01, 0.0, -0.0268]},
                {"half_size": [0.01, 0.0209, 0.05], "pos": [-0.01, 0.0291, 0.0]},
                {"half_size": [0.01, 0.0209, 0.05], "pos": [-0.01, -0.0291, 0.0]},
                {"half_size": [0.01, 0.0058, 0.0037], "pos": [-0.01, 0.0, 0.0]},
            ], 0.3539, [1, 1, 1], (0.0598, 0.0905, 0.10), True),
        ],
        "success": _moved("charger", (-0.0496, 0.1661)),
    },
}
