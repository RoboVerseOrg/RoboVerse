# Parity and consistency probes

Scripts that measure how far a task or backend is from a reference: an upstream framework
(mjlab, ManiSkill, LIBERO, SIMPLER), another MetaSim backend, or a recorded policy rollout. They
**report** numbers and exit non-zero on failure; the contracts that must hold on every PR live in
`tests/` and `packages/metasim/metasim/test/sim/test_replay.py`.

| Script | Compares |
|---|---|
| `parity_superdex_tracking.py` | SuperDex vs MuJoCo joint tracking / object drop on the same actions |
| `eval_cartpole_cross_sim.py`, `eval_go1_cross_sim.py`, `parity_go1_diag.py`, `parity_obs_reward_cartpole.py`, `render_cartpole_rollout.py` | mjlab-ported tasks across backends and against mjlab observations / rewards |
| `parity_simpler_env.py`, `parity_policy_simpler_env.py`, `policy_rollout_simpler_env.py`, `trajectory_simpler_env.py`, `render_simpler_env.py`, `gen_simpler_doc_gallery.py` | SIMPLER (SAPIEN) tasks vs their MetaSim ports, with and without a policy |
| `parity_liberoplus_passthrough.py`, `eval_liberoplus_policy_consistency.py`, `gen_libero_sidebyside.py` | LIBERO+ passthrough vs native MetaSim replay |
| `spike_metasim_full_parity.py`, `verify_native_registration.py`, `render_native_gallery.py`, `render_policy_gallery.py`, `render_metasim_1to1_gallery.py` | full-suite registration / rollout / gallery checks |
| `test_mjlab_v2_backward_compat.py` | smoke over the 12 mjlab v2 tasks on MuJoCo and Newton |

Run from the repository root: `python tools/parity/<script>.py --help`. Each script names the
extra environment it needs (SAPIEN, LIBERO, a GPU for Newton) at the top of its docstring.
