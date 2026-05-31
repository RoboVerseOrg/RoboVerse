"""Rough-terrain actor-observation parity: native mjlab vs RoboVerse v2 ports.

Closes the last obs-parity gap for the rough velocity tasks
(``mjlab.velocity_rough_go1_v2`` / ``mjlab.velocity_rough_g1_v2``): the actor obs
is the flat proprio vector (48-D go1 / 99-D g1) plus a trailing ``height_scan(187)``
term from the yaw-aligned terrain grid raycast (mjlab ``terrain_scan``,
``GridPatternCfg(size=(1.6, 1.0), resolution=0.1)`` → 17x11 = 187 rays,
``ray_alignment="yaw"``, ``max_distance=5.0``, obs ``scale = 1/max_distance = 0.2``;
``mjlab/tasks/velocity/velocity_env_cfg.py:44-54,200-205``).

Two checks per task:

  (A) FLAT-PROPRIO PARITY (native mjlab env, state-injection): inject K identical
      matched states (root + joint + command + last_action) into BOTH the native
      mjlab env and the RV env and compare the 48-/99-D proprio prefix term-by-term.
      Target: bitwise / float-eps. (This is the already-passing flat obs reused by
      the rough cfg.)

  (B) HEIGHT_SCAN PARITY (independent mjlab-algorithm ground truth): the native
      mjlab height_scan reads a mujoco_warp BVH raycast that is captured in a CUDA
      graph at env init and CANNOT be re-driven by a manual ``write_root_state`` +
      ``forward`` outside the step loop (the sensor stays frozen at the reset pose;
      verified empirically). So instead of injecting into the native env, we verify
      the RV sensor against an INDEPENDENT ground truth computed with mjlab's OWN
      source functions — ``GridPatternCfg.generate_rays`` +
      ``RayCastSensor._extract_yaw_rotation`` + plain ``mujoco.mj_ray`` — on the
      SAME RV ``mj_model`` / ``mj_data`` at the injected pose. This validates the RV
      port reproduces mjlab's exact grid pattern, yaw alignment, height read
      (``frame_z - hit_z``) and clip. We exercise it on (i) the flat ground plane
      and (ii) a deterministic tilted base (the yaw-rotated grid samples a different
      xy footprint per ray), and we ALSO confirm the native mjlab env's own sensor
      tracks correctly when the env is stepped normally (sanity).

Matched terrain (documented):
  Both sides sit on a FLAT GROUND PLANE at z=0 (RV's mujoco scene already carries a
  group-0 ground plane; the native rough cfg's procedural terrain is switched to
  ``terrain_type="plane"``). With identical poses the grid rays read identical
  heights, isolating the SENSOR (grid + yaw + height read + clip) — which is what
  we verify. Reproducing mjlab's full procedural terrain generator 1:1 is OUT OF
  SCOPE (a separate, larger port; see roboverse_pack/tasks/mjlab/mdp/terrain.py).

Run:
    MJLAB_REPO=/workspace/mjlab_upstream PYTHONPATH=$PWD:$PWD/scripts MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES=0 python scripts/parity_obs_rough.py
"""

from __future__ import annotations

import numpy as np
import torch

from parity_obs_all import (  # noqa: E402  (sibling-script reuse)
    build_mjlab,
    build_rv,
    mjlab_inject,
    native_compute_actor,
    native_term_layout,
    rv_compute_actor,
    rv_inject,
    rv_term_layout,
)

K = 16
SEED = 0
DEVICE = "cuda:0"
TOL = 1e-5


def _go1_rough_plane_builder(play=False):
    """Native go1 rough cfg with terrain switched to a flat plane at z=0."""
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg

    cfg = unitree_go1_rough_env_cfg(play=play)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 512
    return cfg


def _g1_rough_plane_builder(play=False):
    """Native g1 rough cfg with terrain switched to a flat plane at z=0."""
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

    cfg = unitree_g1_rough_env_cfg(play=play)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 512
    return cfg


# ---------------------------------------------------------------------------
# mjlab-algorithm ground truth for height_scan, computed with mjlab's OWN source
# functions on a plain mujoco model/data (no CUDA graph).
# ---------------------------------------------------------------------------
def _mjlab_height_scan_groundtruth(mp, data, *, frame_body_id, geom_groups, max_distance, scale):
    """Compute height_scan exactly the way mjlab does, via its source functions.

    Uses ``GridPatternCfg.generate_rays`` (raycast_sensor.py:65) for the local
    grid, ``RayCastSensor._extract_yaw_rotation`` (raycast_sensor.py:826) for the
    yaw alignment, and ``mujoco.mj_ray`` for the hit. Returns the scaled
    ``(frame_z - hit_z)`` per ray with misses -> ``max_distance``.
    """
    import mujoco

    from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensor, RayCastSensorCfg

    grid = GridPatternCfg(size=(1.6, 1.0), resolution=0.1)
    local_off, local_dir = grid.generate_rays(None, "cpu")
    local_off = local_off.numpy().astype(np.float64)
    local_dir = local_dir.numpy().astype(np.float64)
    R = local_off.shape[0]

    frame_pos = np.asarray(data.xpos[frame_body_id], dtype=np.float64)
    frame_mat = np.asarray(data.xmat[frame_body_id], dtype=np.float64).reshape(3, 3)

    # mjlab yaw extraction via its own method (on a throwaway sensor instance).
    sensor = RayCastSensor(RayCastSensorCfg(name="_gt", frame=(), ray_alignment="yaw", pattern=grid))
    rot = sensor._extract_yaw_rotation(torch.from_numpy(frame_mat).unsqueeze(0).float())[0].numpy().astype(np.float64)

    geomgroup = np.zeros(6, dtype=np.uint8)
    for g in geom_groups:
        if 0 <= g < 6:
            geomgroup[g] = 1
    geomid_out = np.zeros(1, dtype=np.int32)

    heights = np.full(R, max_distance, dtype=np.float64)
    for r in range(R):
        origin = frame_pos + rot @ local_off[r]
        world_dir = rot @ local_dir[r]
        d = mujoco.mj_ray(mp, data.ptr, origin, world_dir, geomgroup, 1, -1, geomid_out)
        if d < 0 or d > max_distance:
            heights[r] = max_distance
        else:
            hit_z = origin[2] + world_dir[2] * d
            heights[r] = frame_pos[2] - hit_z
    return (heights * scale).astype(np.float32)


def _force_rv_state(rv, *, root, jpos, jvel):
    """Write a matched root + joint state into the RV physics and forward-kinematics."""
    rv_inject(
        rv, root=root, jpos=jpos, jvel=jvel, command={"twist": [0.0, 0.0, 0.0]},
        last_action=np.zeros(len(jpos)), n_qpos_joint=len(jpos), free_base=True,
    )
    import mujoco

    ph = rv.handler.physics
    mp = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
    mujoco.mj_forward(mp, ph.data.ptr)


def _sample_root_joints(rng, n_joints, *, base_z, tilt):
    pos = np.array(
        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), base_z + rng.uniform(-0.05, 0.15)],
        dtype=np.float64,
    )
    if tilt:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-9
        angle = rng.uniform(-0.3, 0.3)
        quat = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)], dtype=np.float64)
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    lin = rng.uniform(-0.5, 0.5, size=3)
    ang = rng.uniform(-0.5, 0.5, size=3)
    root = np.concatenate([pos, quat, lin, ang]).astype(np.float64)
    jpos = rng.uniform(-0.4, 0.4, size=n_joints).astype(np.float64)
    jvel = rng.uniform(-0.5, 0.5, size=n_joints).astype(np.float64)
    return root, jpos, jvel


# ---------------------------------------------------------------------------
# (A) flat-proprio parity vs the native env (state injection).
# ---------------------------------------------------------------------------
def proprio_parity(name, native_builder, rv_task, n_joints, base_z, rng):
    mj = build_mjlab(native_builder)
    rv = build_rv(rv_task)
    mj_layout = native_term_layout(mj)
    rv_layout = rv_term_layout(rv)
    # proprio prefix = everything before height_scan (the last term).
    prefix = mj_layout[:-1]
    prefix_dim = sum(d for _, d in prefix)

    per_term = {}
    full_prefix_max = 0.0
    for _ in range(K):
        root, jpos, jvel = _sample_root_joints(rng, n_joints, base_z=base_z, tilt=True)
        cmd = {"twist": list(rng.uniform(-1.0, 1.0, size=3))}
        act = rng.uniform(-1.0, 1.0, size=n_joints).astype(np.float64)
        mjlab_inject(mj, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act, ent_name="robot")
        rv_inject(rv, root=root, jpos=jpos, jvel=jvel, command=cmd, last_action=act,
                  n_qpos_joint=n_joints, free_base=True)
        mo = native_compute_actor(mj)
        ro = rv_compute_actor(rv)
        off = 0
        for n, d in prefix:
            md = float(np.abs(mo[off : off + d] - ro[off : off + d]).max())
            per_term[n] = max(per_term.get(n, 0.0), md)
            off += d
        full_prefix_max = max(full_prefix_max, float(np.abs(mo[:prefix_dim] - ro[:prefix_dim]).max()))
    native_dim = len(mo)
    return dict(per_term=per_term, prefix_max=full_prefix_max, native_dim=native_dim, prefix_dim=prefix_dim,
                hs_native_dim=mj_layout[-1][1], hs_native_off=prefix_dim)


# ---------------------------------------------------------------------------
# (B) height_scan parity vs mjlab-algorithm ground truth on the RV physics.
# ---------------------------------------------------------------------------
def height_scan_parity(rv_task, n_joints, frame_body, base_z, rng, scale=0.2):
    import mujoco

    rv = build_rv(rv_task)
    rv_layout = rv_term_layout(rv)
    hs_off = sum(d for _, d in rv_layout[:-1])
    hs_dim = rv_layout[-1][1]

    ph = rv.handler.physics
    mp = ph.model.ptr if hasattr(ph.model, "ptr") else ph.model
    fbid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, frame_body)

    flat_max = 0.0
    tilt_max = 0.0
    for tilt, store in ((False, "flat"), (True, "tilt")):
        m = 0.0
        for _ in range(K):
            root, jpos, jvel = _sample_root_joints(rng, n_joints, base_z=base_z, tilt=tilt)
            _force_rv_state(rv, root=root, jpos=jpos, jvel=jvel)
            ro = rv_compute_actor(rv)  # triggers sensor.update()
            rv_hs = ro[hs_off : hs_off + hs_dim]
            gt = _mjlab_height_scan_groundtruth(
                mp, ph.data, frame_body_id=fbid, geom_groups=(0,), max_distance=5.0, scale=scale
            )
            m = max(m, float(np.abs(rv_hs - gt).max()))
        if store == "flat":
            flat_max = m
        else:
            tilt_max = m
    return dict(hs_dim=hs_dim, flat_max=flat_max, tilt_max=tilt_max)


def native_sensor_sanity(native_builder):
    """Confirm the native env's own height_scan tracks the base when stepped."""
    mj = build_mjlab(native_builder)
    ts = mj.scene["terrain_scan"]
    for _ in range(5):
        mj.step(mj.action_manager.action * 0.0)
    mo = native_compute_actor(mj)
    frame_z = float(ts.data.frame_pos_w[0, 0, 2])
    hs0 = float(mo[-187])
    return abs(hs0 - 0.2 * frame_z)


def run(name, native_builder, rv_task, n_joints, frame_body, base_z, rng):
    a = proprio_parity(name, native_builder, rv_task, n_joints, base_z, rng)
    b = height_scan_parity(rv_task, n_joints, frame_body, base_z, rng)
    san = native_sensor_sanity(native_builder)
    hs_max = max(b["flat_max"], b["tilt_max"])
    passed = (a["prefix_max"] <= TOL) and (hs_max <= TOL)
    return dict(
        task=name, native_dim=a["native_dim"], hs_dim=b["hs_dim"],
        proprio_max=a["prefix_max"], per_term=a["per_term"],
        hs_flat=b["flat_max"], hs_tilt=b["tilt_max"], hs_max=hs_max,
        native_sanity=san, passed=passed,
    )


def main():
    rng = np.random.default_rng(SEED)
    results = [
        run("mjlab.velocity_rough_go1_v2", _go1_rough_plane_builder, "mjlab.velocity_rough_go1_v2", 12, "trunk", 0.40, rng),
        run("mjlab.velocity_rough_g1_v2", _g1_rough_plane_builder, "mjlab.velocity_rough_g1_v2", 29, "pelvis", 0.74, rng),
    ]

    print("\n" + "=" * 104)
    print("ROUGH-TERRAIN ACTOR-OBS PARITY (matched flat ground plane at z=0)")
    print("=" * 104)
    hdr = (
        f"{'task':<34}{'native_dim':>11}{'hs_dim':>8}{'max|d|hs':>12}"
        f"{'max|d|proprio':>15}{'result':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    all_pass = True
    for r in results:
        all_pass = all_pass and r["passed"]
        print(
            f"{r['task']:<34}{r['native_dim']:>11}{r['hs_dim']:>8}"
            f"{r['hs_max']:>12.3e}{r['proprio_max']:>15.3e}"
            f"{('PASS' if r['passed'] else 'FAIL'):>9}"
        )
    print("=" * 104)
    for r in results:
        print(f"\n{r['task']}:")
        print(f"    height_scan max|d|  flat={r['hs_flat']:.3e}  tilted={r['hs_tilt']:.3e}")
        print(f"    native-env sensor sanity (|hs0 - 0.2*frame_z| after 5 steps) = {r['native_sanity']:.3e}")
        print("    proprio per-term max|delta|:")
        for n, v in r["per_term"].items():
            print(f"        {n:<22} {v:.3e}")
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
