"""Action-level 1:1 parity for multi-agent ManiSkill tasks (e.g. TwoRobot*).

Multi-agent envs expose ``agent`` as a ``MultiAgent`` (no single ``.robot``) and a Dict action space
(one ``pd_joint_delta_pos`` action per robot). This captures every robot + the manipuland, rebuilds
them in one clean SAPIEN3 scene via the reproduction recipe, replays the per-robot action slices, and
reports the object-pose delta vs native — proving the action contract maps 1:1 for the multi-robot
case too.

    SAPIEN_HEADLESS=1 python -m tools.maniskill_integration.parity_multi_agent --task TwoRobotPickCube-v1
"""

from __future__ import annotations

import argparse

import numpy as np

from . import recipe as R


def _sim_spec(scfg):
    sc, dm = scfg.scene_config, scfg.default_materials_config
    return R.SimSpec(
        sim_freq=scfg.sim_freq,
        control_freq=scfg.control_freq,
        gravity=np.array([0, 0, -9.81]),
        bounce_threshold=sc.bounce_threshold,
        sleep_threshold=sc.sleep_threshold,
        contact_offset=sc.contact_offset,
        rest_offset=sc.rest_offset,
        solver_position_iterations=sc.solver_position_iterations,
        solver_velocity_iterations=sc.solver_velocity_iterations,
        enable_pcm=sc.enable_pcm,
        enable_tgs=sc.enable_tgs,
        enable_ccd=sc.enable_ccd,
        enable_enhanced_determinism=sc.enable_enhanced_determinism,
        enable_friction_every_iteration=sc.enable_friction_every_iteration,
        cpu_workers=sc.cpu_workers,
        mat_static_friction=dm.static_friction,
        mat_dynamic_friction=dm.dynamic_friction,
        mat_restitution=dm.restitution,
    )


def run(task_id: str, steps: int = 20, seed: int = 0, obj_name: str = "cube") -> dict:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    import sapien.physx as physx
    import torch
    from transforms3d.euler import euler2quat

    from roboverse_pack.tasks.maniskill._native.control import PDJointDeltaPos

    rng = np.random.RandomState(seed + 1)
    env = gym.make(task_id, num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    u = env.unwrapped
    env.reset(seed=seed)
    agents = u.agent.agents
    keys = list(u.agent.action_space.spaces.keys())
    per = u.agent.action_space.spaces[keys[0]].shape[-1]

    def cap_robot(rb):
        aj = rb._objs[0].get_active_joints()
        rp = rb.get_pose().raw_pose.cpu().numpy().ravel()
        return dict(
            rp=rp,
            qpos=rb.get_qpos().cpu().numpy().ravel().astype(np.float64),
            qvel=rb.get_qvel().cpu().numpy().ravel().astype(np.float64),
            stiff=np.array([float(np.asarray(j.stiffness).ravel()[0]) for j in aj]),
            damp=np.array([float(np.asarray(j.damping).ravel()[0]) for j in aj]),
            fl=np.array([float(np.asarray(j.force_limit).ravel()[0]) for j in aj]),
        )

    robos = [cap_robot(a.robot) for a in agents]
    urdfs = [a.urdf_path for a in agents]
    obj = u.scene.actors[obj_name]
    obj0 = obj.pose.raw_pose.cpu().numpy().ravel()
    hs = float(np.array(u.cube_half_size).ravel()[0])
    cmass = float(np.asarray(obj.mass).ravel()[0])
    sim = _sim_spec(u.sim_config)

    actions = rng.uniform(-1, 1, size=(steps, len(keys) * per)).astype(np.float32)
    ref = []
    for a in actions:
        act = {k: torch.tensor(a[i * per : (i + 1) * per]).unsqueeze(0) for i, k in enumerate(keys)}
        env.step(act)
        ref.append(obj.pose.p.cpu().numpy().ravel().copy())
    env.close()
    ref = np.asarray(ref)

    # --- replica: N robots + object on the recipe table ---
    R.apply_global_physx(sim)
    scene = sapien.Scene([physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
    scene.set_timestep(1.0 / sim.sim_freq)
    scene.add_ground(-R.TABLE_HEIGHT)
    mat0 = physx.PhysxMaterial(sim.mat_static_friction, sim.mat_dynamic_friction, sim.mat_restitution)
    tb = scene.create_actor_builder()
    tb.add_box_collision(pose=sapien.Pose(p=list(R.TABLE_LOCAL_OFFSET_P)), half_size=list(R.TABLE_HALF), material=mat0)
    table = tb.build_kinematic(name="table-workspace")
    table.set_pose(sapien.Pose(p=list(R.TABLE_INITIAL_P), q=euler2quat(0, 0, np.pi / 2)))

    robots = []
    for i, r in enumerate(robos):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        rob = loader.load(urdfs[i])
        rob.set_root_pose(sapien.Pose(r["rp"][:3], r["rp"][3:]))
        rob.set_qpos(r["qpos"].astype(np.float32))
        rob.set_qvel(r["qvel"].astype(np.float32))
        for k, j in enumerate(rob.get_active_joints()):
            j.set_drive_property(
                stiffness=float(r["stiff"][k]),
                damping=float(r["damp"][k]),
                force_limit=float(r["fl"][k]),
                mode="force",
            )
        for link in rob.get_links():
            link.disable_gravity = True
        robots.append(rob)

    bd = scene.create_actor_builder()
    bd.add_box_collision(half_size=[hs] * 3, material=mat0)
    cb = bd.build(name=obj_name)
    cb.set_pose(sapien.Pose(obj0[:3], obj0[3:]))
    for c in cb.get_components():
        if isinstance(c, physx.PhysxRigidDynamicComponent):
            c.mass = cmass

    # each panda arm+gripper = 8-dim; arm-only would be 7 (detect by active joints).
    controllers = [
        PDJointDeltaPos(arm_dof=per - 1, gripper=(len(rob.get_active_joints()) == per + 1)) for rob in robots
    ]
    rep = []
    for a in actions:
        for i, rob in enumerate(robots):
            aj = rob.get_active_joints()
            q = rob.get_qpos().astype(np.float32).copy()
            tg = controllers[i].compute_targets(q, a[i * per : (i + 1) * per])
            for k, j in enumerate(aj):
                j.set_drive_target(float(tg[k]))
        for _ in range(sim.decimation):
            scene.step()
        rep.append(cb.get_pose().p.copy())
    rep = np.asarray(rep)
    delta = float(np.abs(ref - rep).max())
    return {"task": task_id, "n_robots": len(robots), "per_robot_action": per, "obj_pose_delta_max": delta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="TwoRobotPickCube-v1")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obj", default="cube")
    args = ap.parse_args(argv)
    r = run(args.task, args.steps, args.seed, args.obj)
    print(
        f"{r['task']}: {r['n_robots']} robots x {r['per_robot_action']}-dim — obj pose Δmax={r['obj_pose_delta_max']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
