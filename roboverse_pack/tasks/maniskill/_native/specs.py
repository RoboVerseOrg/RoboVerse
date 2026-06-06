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

import numpy as np

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


def _pull_cube_reset(task, rng):
    """ManiSkill PullCube reset: cube XY ∈ U(±0.1); goal = cube - [0.2, 0, 0] (goal_radius 0.1)."""
    cx, cy = rng.uniform(-0.1, 0.1, 2)
    _set_cube(task, "cube", cx, cy, 0.02)
    return [cx - 0.2, cy, 1e-3]


def _pull_cube_reward(task):
    return RW.pull_cube(cube_pos=task.obj_pos("cube"), tcp_pos=task.tcp_pos(), goal_pos=task.goal_pos)


def _pull_cube_success(task):
    return SU.pull_cube(cube_pos=task.obj_pos("cube"), goal_pos=task.goal_pos)


def _poke_cube_reset(task, rng):
    """ManiSkill PokeCube reset: peg XY ∈ U(±0.1); cube x = peg_x + peg_half_len(0.12)+0.1; goal = cube + [0.05+0.05,0,0]."""
    import sapien

    px, py = rng.uniform(-0.1, 0.1, 2)
    task.handler.object_ids["peg"].set_pose(sapien.Pose([float(px), float(py), 0.025], [1, 0, 0, 0]))
    cx = px + 0.12 + 0.1
    cy = float(rng.uniform(-0.1, 0.1))
    _set_cube(task, "cube", cx, cy, 0.02)
    return [cx + 0.1, cy, 1e-3]  # 0.05 + goal_radius 0.05


def _poke_cube_success(task):
    return SU.poke_cube(cube_pos=task.obj_pos("cube"), goal_pos=task.goal_pos,
                        is_robot_static=task.is_robot_static())


def _lift_peg_reset(task, rng):
    """ManiSkill LiftPegUpright reset: peg XY ∈ U(±0.1), z = peg_half_width(0.025), lying (euler(π/2,0,0))."""
    import sapien
    from transforms3d.euler import euler2quat

    px, py = rng.uniform(-0.1, 0.1, 2)
    q = euler2quat(np.pi / 2, 0, 0)
    task.handler.object_ids["peg"].set_pose(sapien.Pose([float(px), float(py), 0.025], [float(v) for v in q]))
    return [0.0, 0.0, 0.0]


def _lift_peg_reward(task):
    w, x, y, z = task.obj_quat("peg")
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return RW.lift_peg_upright(peg_rot_mat=R, peg_pos=task.obj_pos("peg"), tcp_pos=task.tcp_pos(),
                               is_grasped=task.is_grasped("peg"), peg_half_length=0.12)


def _lift_peg_success(task):
    return SU.lift_peg_upright(peg_euler_z=task.obj_euler_z("peg"),
                              peg_z=float(task.obj_pos("peg")[2]), peg_half_length=0.12)


def _stack_cube_reset(task, rng):
    """ManiSkill StackCube reset: cubeA/cubeB XY in [-0.1,0.1]x[-0.2,0.2] (collision-avoided)."""
    import sapien

    base = rng.uniform(-0.1, 0.1, 2)
    a = base + rng.uniform([-0.1, -0.2], [0.1, 0.2])
    b = base + rng.uniform([-0.1, -0.2], [0.1, 0.2])
    task.handler.object_ids["cubeA"].set_pose(sapien.Pose([float(a[0]), float(a[1]), 0.02], [1, 0, 0, 0]))
    task.handler.object_ids["cubeB"].set_pose(sapien.Pose([float(b[0]), float(b[1]), 0.02], [1, 0, 0, 0]))
    return [0.0, 0.0, 0.0]  # stack has no goal site


def _on_cube_geom(a, b, hs=(0.02, 0.02, 0.02)):
    """Geometric is-A-on-B (ManiSkill xy + z flags), without the static/grasp parts."""
    import numpy as _np

    off = _np.asarray(a, _np.float64) - _np.asarray(b, _np.float64)
    xy = _np.linalg.norm(off[:2]) <= _np.linalg.norm(_np.asarray(hs)[:2]) + 0.005
    z = abs(off[2] - hs[2] * 2) <= 0.005
    return bool(xy and z)


def _stack_cube_reward(task):
    A, B = task.obj_pos("cubeA"), task.obj_pos("cubeB")
    grasped = task.is_grasped("cubeA")
    on = _on_cube_geom(A, B)
    success = on and task.obj_is_static("cubeA") and not grasped
    return RW.stack_cube(
        tcp_pos=task.tcp_pos(), cubeA_pos=A, cubeB_pos=B, cube_half_size_z=0.02,
        finger_qpos=task.finger_qpos(), gripper_width=task.GRIPPER_WIDTH,
        cubeA_linvel=task.obj_linvel("cubeA"), cubeA_angvel=task.obj_angvel("cubeA"),
        is_cubeA_grasped=grasped, is_cubeA_on_cubeB=on, success=success)


def _stack_cube_success(task):
    return SU.stack_cube(
        cubeA_pos=task.obj_pos("cubeA"), cubeB_pos=task.obj_pos("cubeB"), cube_half_size=[0.02, 0.02, 0.02],
        is_cubeA_static=task.obj_is_static("cubeA"), is_cubeA_grasped=task.is_grasped("cubeA"))


def _place_sphere_reset(task, rng):
    """ManiSkill PlaceSphere reset: sphere x∈[-0.1,-0.05] y∈[-0.1,0.1]; bin x∈[0,0.1] y∈[-0.1,0.1]."""
    import sapien

    sx = float(rng.uniform(-0.1, -0.05)); sy = float(rng.uniform(-0.1, 0.1))
    task.handler.object_ids["sphere"].set_pose(sapien.Pose([sx, sy, 0.02], [1, 0, 0, 0]))
    bx = float(rng.uniform(0.0, 0.1)); by = float(rng.uniform(-0.1, 0.1))
    task.handler.object_ids["bin"].set_pose(sapien.Pose([bx, by, 0.0025], [1, 0, 0, 0]))
    return [0.0, 0.0, 0.0]


def _on_bin_geom(obj, bin_pos, radius=0.02, block0=0.025):
    import numpy as _np

    off = _np.asarray(obj, _np.float64) - _np.asarray(bin_pos, _np.float64)
    return bool(_np.linalg.norm(off[:2]) <= 0.005 and abs(off[2] - radius - block0) <= 0.005)


def _place_sphere_reward(task):
    obj, binp = task.obj_pos("sphere"), task.obj_pos("bin")
    grasped = task.is_grasped("sphere")
    on_bin = _on_bin_geom(obj, binp)
    success = on_bin and task.obj_is_static("sphere") and not grasped
    return RW.place_sphere(
        tcp_pos=task.tcp_pos(), obj_pos=obj, bin_pos=binp, block_half_size0=0.025, radius=0.02,
        finger_qpos=task.finger_qpos(), gripper_width=task.GRIPPER_WIDTH,
        obj_linvel=task.obj_linvel("sphere"), obj_angvel=task.obj_angvel("sphere"),
        robot_is_static=task.is_robot_static(), is_obj_grasped=grasped, is_obj_on_bin=on_bin, success=success)


def _place_sphere_success(task):
    return SU.place_sphere(
        obj_pos=task.obj_pos("sphere"), bin_pos=task.obj_pos("bin"), radius=0.02, block_half_size0=0.025,
        is_obj_static=task.obj_is_static("sphere"), is_obj_grasped=task.is_grasped("sphere"))


def _roll_ball_reset(task, rng):
    """ManiSkill RollBall reset: ball x∈[-0.4,0.2] y∈[0.5,0.7]; goal x∈[-0.4,0.2] y∈[-1.0,-0.8]."""
    import sapien

    bx = float(rng.uniform(-1, 1) * 0.3 - 0.1); by = float(rng.uniform(0.5, 0.7))
    task.handler.object_ids["ball"].set_pose(sapien.Pose([bx, by, 0.035], [1, 0, 0, 0]))
    task._roll_reached = 0.0
    gx = float(rng.uniform(-1, 1) * 0.3 - 0.1); gy = float(rng.uniform(0.5, 0.7) - 1.5 + 0.1)
    return [gx, gy, 1e-3]


def _roll_ball_reward(task):
    r, task._roll_reached = RW.roll_ball(
        tcp_pos=task.tcp_pos(), ball_pos=task.obj_pos("ball"), goal_pos=task.goal_pos,
        ball_radius=0.035, reached_status=getattr(task, "_roll_reached", 0.0), success=False)
    return r


def _roll_ball_success(task):
    return SU.roll_ball(ball_pos=task.obj_pos("ball"), goal_pos=task.goal_pos)


def _pull_cube_tool_reset(task, rng):
    """ManiSkill PullCubeTool reset: tool XY ∈ [-0.3,-0.1]; cube x = arm_reach+rand*0.2-0.3, y ∈ ±0.15."""
    import sapien

    tx = float(-rng.uniform(0, 0.2) - 0.1); ty = float(-rng.uniform(0, 0.2) - 0.1)
    task.handler.object_ids["l_shape_tool"].set_pose(sapien.Pose([tx, ty, 0.025], [1, 0, 0, 0]))
    cx = 0.35 + float(rng.uniform(0, 0.2)) - 0.3
    cy = float(rng.uniform(0, 0.3) - 0.15)
    _set_cube(task, "cube", cx, cy, 0.02)
    return [0.0, 0.0, 0.0]


def _pull_cube_tool_reward(task):
    return RW.pull_cube_tool(
        tcp_pos=task.tcp_pos(), cube_pos=task.obj_pos("cube"), tool_pos=task.obj_pos("l_shape_tool"),
        robot_base_pos=task.robot_base_pos(), is_grasping=task.is_grasped("l_shape_tool", max_angle=20),
        success=SU.pull_cube_tool(cube_pos=task.obj_pos("cube"), robot_base_pos=task.robot_base_pos()),
        hook_length=0.05, cube_half_size=0.02, arm_reach=0.35, cube_size=0.02)


def _pull_cube_tool_success(task):
    return SU.pull_cube_tool(cube_pos=task.obj_pos("cube"), robot_base_pos=task.robot_base_pos())


def _stack_pyramid_reset(task, rng):
    """ManiSkill StackPyramid reset: 3 cubes in [-0.1,0.1]x[-0.2,0.2] (collision-avoided)."""
    import sapien

    base = rng.uniform(-0.1, 0.1, 2)
    for n in ("cubeA", "cubeB", "cubeC"):
        xy = base + rng.uniform([-0.1, -0.2], [0.1, 0.2])
        task.handler.object_ids[n].set_pose(sapien.Pose([float(xy[0]), float(xy[1]), 0.02], [1, 0, 0, 0]))
    return [0.0, 0.0, 0.0]


def _stack_pyramid_success(task):
    P = {n: task.obj_pos(n) for n in ("cubeA", "cubeB", "cubeC")}
    statics = {k: task.obj_is_static("cube" + k) for k in ("A", "B", "C")}
    grasps = {k: task.is_grasped("cube" + k) for k in ("A", "B", "C")}
    return SU.stack_pyramid(posA=P["cubeA"], posB=P["cubeB"], posC=P["cubeC"],
                            cube_half_size=[0.02, 0.02, 0.02], statics=statics, grasps=grasps)


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
        "goal": _pull_cube_reset, "reward": _pull_cube_reward, "success_full": _pull_cube_success,
    },
    "stack_cube": {
        "gym_id": "StackCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cubeA", "box", _CUBE, 0.064, _RED, (-0.0831, -0.0935, 0.02), False),
            ("cubeB", "box", _CUBE, 0.064, _GREEN, (-0.0393, 0.1073, 0.02), False),
        ],
        "success": _stacked("cubeA", "cubeB"),
        "goal": _stack_cube_reset, "reward": _stack_cube_reward, "success_full": _stack_cube_success,
    },
    "poke_cube": {
        "gym_id": "PokeCube-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cube", "box", _CUBE, 0.064, _RED, (0.2193, -0.0385, 0.02), False),
            ("peg", "box", [0.24, 0.05, 0.05], 0.6, _MSBLUE, (-0.0007, 0.0536, 0.025), False),
        ],
        "success": _moved("cube", (0.2193, -0.0385)),
        "goal": _poke_cube_reset, "success_full": _poke_cube_success,
    },
    "lift_peg_upright": {
        "gym_id": "LiftPegUpright-v1", "base": _BASE, "max_steps": 50,
        "objects": [("peg", "box", [0.24, 0.05, 0.05], 0.6, [0.69, 0.055, 0.055], (-0.0007, 0.0536, 0.025), False)],
        "success": _lifted("peg", 0.1),
        "goal": _lift_peg_reset, "reward": _lift_peg_reward, "success_full": _lift_peg_success,
    },
    "roll_ball": {
        "gym_id": "RollBall-v1", "base": ((-0.10, 1.0, 0.0), (0.7071068, 0.0, 0.0, -0.7071068)),
        "max_steps": 80,
        "objects": [("ball", "sphere", 0.035, 0.17959, _BLUE, (-0.1022, 0.6536, 0.035), False)],
        "success": _moved("ball", (-0.1022, 0.6536), 0.1),
        "goal": _roll_ball_reset, "reward": _roll_ball_reward, "success_full": _roll_ball_success,
    },
    "place_sphere": {
        "gym_id": "PlaceSphere-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("sphere", "sphere", 0.02, 0.03351, _MSBLUE, (-0.0752, 0.0536, 0.02), False),
            ("bin", "box", [0.05, 0.05, 0.005], 0.0225, [1, 1, 1], (0.0088, -0.0736, 0.0025), True),
        ],
        "success": _moved("sphere", (-0.0752, 0.0536)),
        "goal": _place_sphere_reset, "reward": _place_sphere_reward, "success_full": _place_sphere_success,
    },
    "stack_pyramid": {
        "gym_id": "StackPyramid-v1", "base": _BASE, "max_steps": 50,
        "objects": [
            ("cubeA", "box", _CUBE, 0.064, _RED, (-0.0007, 0.1073, 0.02), False),
            ("cubeB", "box", _CUBE, 0.064, _GREEN, (-0.0823, -0.1472, 0.02), False),
            ("cubeC", "box", _CUBE, 0.064, _BLUE, (-0.0385, 0.0536, 0.02), False),
        ],
        "success": _stacked("cubeC", "cubeA"),
        "goal": _stack_pyramid_reset, "success_full": _stack_pyramid_success,
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
        "goal": _pull_cube_tool_reset, "reward": _pull_cube_tool_reward, "success_full": _pull_cube_tool_success,
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
