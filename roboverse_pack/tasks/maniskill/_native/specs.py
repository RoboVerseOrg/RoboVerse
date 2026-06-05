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


# name -> spec. ``objects``: list of (name, kind, geom, mass, color, pos, kinematic).
#   box geom = full size [x,y,z]; sphere geom = radius.
TASK_SPECS: dict[str, dict] = {
    "pick_cube": {
        "gym_id": "PickCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [("cube", "box", _CUBE, 0.064, _RED, (-0.0007, 0.0536, 0.02), False)],
        "success": _lifted("cube"),
    },
    "push_cube": {
        "gym_id": "PushCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [("cube", "box", _CUBE, 0.064, _MSBLUE, (-0.0007, 0.0536, 0.02), False)],
        "success": _moved("cube", (-0.0007, 0.0536)),
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
}
