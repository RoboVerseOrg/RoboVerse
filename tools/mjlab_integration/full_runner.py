"""Full-pipeline comparison: raw mujoco vs `MujocoHandler.simulate()`.

For each task that we know how to express as a `ScenarioCfg`, build the
scenario, launch the real `MujocoHandler`, drive it for N control steps
(writing ctrl directly to `physics.data.ctrl`), and diff vs the raw rollout.

We keep the scenario builders here (rather than in `roboverse_pack`) to
isolate the comparison harness from the task-registry side of integration.
The registered `mjlab.*` tasks in `roboverse_pack/tasks/mjlab/` are about
making mjlab tasks *usable* from MetaSim, which is a separate concern.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from . import inventory as inv
from .diff import diff_rollouts
from .plot import plot_trajectory_pair
from .raw_rollout import RolloutResult, raw_mujoco_rollout, sinusoid_ctrl, zero_ctrl

_REPORTS = os.environ.get("ROBOVERSE_REPORTS_DIR", "outputs/reports")


def _cartpole_scenario(task: inv.MjlabTask):
    """Load cartpole as the scene MJCF, not as a robot wrap.

    mjlab cartpole.xml is a *complete* scene with its own floor + rails +
    cameras at worldbody. Wrapping it as `RobotCfg(mjcf_path=...)` re-roots
    every worldbody-level geom into a namespace-prefixed attachment body,
    which can shift floors / rails by the wrapper offset and produces
    ground-penetration contacts that the raw mjlab rollout does not see.
    Using `scene=` keeps the original world frame intact.
    """
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.scenario.simulator_params import SimParamCfg

    scene = SceneCfg(mjcf_path=str(task.asset.xml_path))
    scenario = ScenarioCfg(
        scene=scene,
        robots=[],
        sim_params=SimParamCfg(dt=0.01),
        decimation=task.decimation,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,  # cartpole.xml already declares a floor
    )
    return scenario


def _yam_scenario(task: inv.MjlabTask):
    """Wrap the bare YAM robot MJCF as a fixed-base RobotCfg.

    No gripper-finger position targets here — we only diff physics under the
    same applied ctrl (zero by default). We disable MetaSim's auto ground
    plane to match the raw rollout (raw loads only the robot MJCF).
    """
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg

    robot = RobotCfg(
        name="yam",
        num_joints=7,
        fix_base_link=True,
        mjcf_path=str(task.asset.xml_path),
        enabled_gravity=True,
        enabled_self_collisions="mujoco_default",  # match raw mujoco's stock contact filtering (parent-child filtered, others collide); new sentinel added in MetaSim/fix/mujoco-self-collisions-mujoco-default
        control_type={},  # no PD wrap; cfg ctrl directly
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.002),
        decimation=task.decimation,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,  # raw rollout doesn't add a ground either
    )


def _quadruped_scenario(task: inv.MjlabTask, name: str = "go1"):
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg

    robot = RobotCfg(
        name=name,
        num_joints=12,
        fix_base_link=False,
        mjcf_path=str(task.asset.xml_path),
        enabled_gravity=True,
        enabled_self_collisions="mujoco_default",  # match raw mujoco's stock contact filtering (parent-child filtered, others collide); new sentinel added in MetaSim/fix/mujoco-self-collisions-mujoco-default
        control_type={},
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.005),
        decimation=task.decimation,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


def _humanoid_scenario(task: inv.MjlabTask, name: str = "g1"):
    from metasim.scenario.robot import RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg

    robot = RobotCfg(
        name=name,
        num_joints=29,
        fix_base_link=False,
        mjcf_path=str(task.asset.xml_path),
        enabled_gravity=True,
        enabled_self_collisions="mujoco_default",  # match raw mujoco's stock contact filtering (parent-child filtered, others collide); new sentinel added in MetaSim/fix/mujoco-self-collisions-mujoco-default
        control_type={},
    )
    return ScenarioCfg(
        robots=[robot],
        sim_params=SimParamCfg(dt=0.005),
        decimation=task.decimation,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


SCENARIO_BUILDERS: dict[str, Callable[[inv.MjlabTask], object]] = {
    "mjlab.cartpole_balance": _cartpole_scenario,
    "mjlab.cartpole_swingup": _cartpole_scenario,
    "mjlab.lift_cube_yam": _yam_scenario,
    "mjlab.lift_cube_yam_rgb": _yam_scenario,
    "mjlab.lift_cube_yam_depth": _yam_scenario,
    "mjlab.multi_cube_seg_yam": _yam_scenario,
    "mjlab.velocity_flat_go1": _quadruped_scenario,
    "mjlab.velocity_rough_go1": _quadruped_scenario,
    "mjlab.velocity_flat_g1": _humanoid_scenario,
    "mjlab.velocity_rough_g1": _humanoid_scenario,
    "mjlab.tracking_flat_g1": _humanoid_scenario,
    "mjlab.tracking_flat_g1_no_state_est": _humanoid_scenario,
}


def _ctrl_for(asset_name: str, nu: int, dt: float):
    if asset_name == "cartpole":
        return sinusoid_ctrl(nu=nu, amplitude=0.3, freq_hz=0.5, dt=dt)
    return zero_ctrl(nu)


def run_full_pipeline_for(task: inv.MjlabTask, out_dir: Path, n_steps: int) -> dict:
    task_dir = out_dir / "runs" / task.metasim_name
    task_dir.mkdir(parents=True, exist_ok=True)

    builder = SCENARIO_BUILDERS.get(task.metasim_name)
    if builder is None:
        return {"task": task.metasim_name, "status": "no_scenario_builder"}

    import mujoco

    probe = mujoco.MjModel.from_xml_path(str(task.asset.xml_path))
    nu = int(probe.nu)
    base_dt = float(probe.opt.timestep)
    ctrl_fn = _ctrl_for(task.asset.name, nu=nu, dt=base_dt)

    summary: dict = {
        "task": task.metasim_name,
        "task_id": task.task_id,
        "asset": task.asset.name,
        "status": "ok",
        "n_steps": n_steps,
    }

    try:
        raw = raw_mujoco_rollout(
            task.asset.xml_path,
            n_steps=n_steps,
            decimation=task.decimation,
            ctrl_fn=ctrl_fn,
            init_joint_pos=task.init_joint_pos,
            task_id=task.metasim_name,
        )
    except Exception as e:
        summary.update(status="raw_failed", error=repr(e), traceback=traceback.format_exc())
        return summary

    try:
        scenario = builder(task)
        scenario.sim_params.dt = base_dt  # match raw mujoco's timestep exactly
        from .metasim_rollout import metasim_full_rollout

        meta_full = metasim_full_rollout(
            scenario,
            n_steps=n_steps,
            decimation=task.decimation,
            ctrl_fn=ctrl_fn,
            init_joint_pos=task.init_joint_pos,
            task_id=task.metasim_name,
        )
    except Exception as e:
        summary.update(status="metasim_full_failed", error=repr(e), traceback=traceback.format_exc())
        return summary

    summary["raw_summary"] = raw.summary()
    summary["metasim_full_summary"] = meta_full.summary()

    # Note: shapes may differ because the MetaSim scenario can introduce
    # extra free joints / extra qpos slots when the original MJCF was bare.
    # We diff only on the joints that are present in both runs by mapping
    # via joint-name suffix match.
    try:
        aligned = _align_by_jointname(raw, meta_full)
        if aligned is None:
            summary["align_warning"] = "could not align joint sets; recording shapes only"
            summary["raw_qpos_shape"] = list(raw.qpos.shape)
            summary["meta_qpos_shape"] = list(meta_full.qpos.shape)
        else:
            raw_aligned, meta_aligned, common_names = aligned
            diff = diff_rollouts(raw_aligned, meta_aligned)
            summary["diff_raw_vs_full"] = diff.as_row()
            summary["aligned_joint_names"] = common_names
            try:
                plot_trajectory_pair(
                    raw_aligned,
                    meta_aligned,
                    task_dir / "qpos_compare_full.png",
                    title=f"{task.metasim_name}  —  raw mujoco vs MetaSim full pipeline",
                )
                summary["full_plot"] = "qpos_compare_full.png"
            except Exception as e:
                summary["full_plot_error"] = repr(e)
    except Exception as e:
        summary.update(diff_error=repr(e), traceback=traceback.format_exc())

    return summary


def _align_by_jointname(a: RolloutResult, b: RolloutResult):
    """Build a pair of RolloutResults restricted to matching joints (by suffix).

    The MetaSim path prefixes joints with `<robotname>/`, so we strip prefix
    via the last "/" segment. We use the recorded `jnt_qposadr` / `jnt_dofadr`
    from each side (populated by the rollout functions) to slice the right
    qpos / qvel windows. Joints contribute a window sized by their joint
    type: free=7q/6v, ball=4q/3v, hinge/slide=1q/1v.
    """
    import mujoco

    def _short(n: str) -> str:
        return n.split("/")[-1]

    a_short = [_short(n) for n in a.joint_names]
    b_short = [_short(n) for n in b.joint_names]
    b_short_to_idx = {n: i for i, n in enumerate(b_short)}
    common = [n for n in a_short if n in b_short_to_idx]
    if not common:
        return None

    if a.jnt_qposadr is None or b.jnt_qposadr is None or a.jnt_type is None or b.jnt_type is None:
        return None  # need the model-derived metadata to slice safely

    def _qslot(jnt_type: int) -> int:
        return {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        }.get(jnt_type, 1)

    def _vslot(jnt_type: int) -> int:
        return {
            int(mujoco.mjtJoint.mjJNT_FREE): 6,
            int(mujoco.mjtJoint.mjJNT_BALL): 3,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        }.get(jnt_type, 1)

    def _gather(rr: RolloutResult, short_to_idx, name) -> tuple[list[int], list[int], int]:
        idx = short_to_idx[name]
        qaddr = rr.jnt_qposadr[idx]
        vaddr = rr.jnt_dofadr[idx]
        jt = rr.jnt_type[idx]
        return (
            list(range(qaddr, qaddr + _qslot(jt))),
            list(range(vaddr, vaddr + _vslot(jt))),
            jt,
        )

    a_short_to_idx = {n: i for i, n in enumerate(a_short)}
    a_q_slots: list[int] = []
    a_v_slots: list[int] = []
    b_q_slots: list[int] = []
    b_v_slots: list[int] = []
    aligned_names: list[str] = []
    for name in common:
        aq, av, ajt = _gather(a, a_short_to_idx, name)
        bq, bv, bjt = _gather(b, b_short_to_idx, name)
        if ajt != bjt or len(aq) != len(bq):
            continue  # joint-type mismatch; skip
        a_q_slots.extend(aq)
        a_v_slots.extend(av)
        b_q_slots.extend(bq)
        b_v_slots.extend(bv)
        aligned_names.append(name)

    if not aligned_names:
        return None

    a_qpos = a.qpos[:, a_q_slots]
    b_qpos = b.qpos[:, b_q_slots]
    a_qvel = a.qvel[:, a_v_slots]
    b_qvel = b.qvel[:, b_v_slots]
    common = aligned_names

    from .raw_rollout import RolloutResult as RR

    a_new = RR(
        backend=a.backend,
        task_id=a.task_id,
        dt=a.dt,
        decimation=a.decimation,
        qpos=a_qpos,
        qvel=a_qvel,
        ctrl=a.ctrl,
        time=a.time,
        joint_names=common,
        actuator_names=a.actuator_names,
    )
    b_new = RR(
        backend=b.backend,
        task_id=b.task_id,
        dt=b.dt,
        decimation=b.decimation,
        qpos=b_qpos,
        qvel=b_qvel,
        ctrl=b.ctrl,
        time=b.time,
        joint_names=common,
        actuator_names=b.actuator_names,
    )
    return a_new, b_new, common


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPORTS, "mjlab_integration"))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--tasks", nargs="*")
    args = p.parse_args(argv)

    out = Path(args.out)
    tasks = inv.ALL_TASKS
    if args.tasks:
        tasks = [t for t in tasks if t.metasim_name in args.tasks]

    summaries = []
    for t in tasks:
        if t.metasim_name not in SCENARIO_BUILDERS:
            continue
        print(f"[full] >> {t.metasim_name}")
        s = run_full_pipeline_for(t, out, n_steps=args.n_steps)
        summaries.append(s)
        if "diff_raw_vs_full" in s:
            d = s["diff_raw_vs_full"]
            print(
                f"   status={s['status']}  qpos_max|Δ|={d.get('qpos_max_abs'):.3e}  qvel_max|Δ|={d.get('qvel_max_abs'):.3e}"
            )
        else:
            print(f"   status={s['status']}  notes={s.get('align_warning') or s.get('error', '')}")

    out_file = out / "summary_full.json"
    out_file.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[full] wrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
