# """Integration tests for metasim/queries/site.py using real MuJoCo and MJX."""

# from __future__ import annotations

# import pytest
# import rootutils
# from loguru import logger as log

# rootutils.setup_root(__file__, pythonpath=True)

# from metasim.queries.site import SitePos, _get_site_id, _site_cache


# def _pick_robot_site_name(handler) -> str:
#     """Pick a site name belonging to the robot from the MuJoCo model."""
#     import pytest as _pytest

#     mj_model = handler.physics.model
#     robot_name = handler.robot.name
#     prefix = f"{robot_name}/"

#     for i in range(mj_model.nsite):
#         name = mj_model.site(i).name
#         if name.startswith(prefix):
#             return name

#     _pytest.skip(f"No site with prefix '{prefix}' found in MuJoCo model")


# def _pick_mjx_robot_site_name(handler) -> str:
#     """Pick a site name belonging to the robot from the MJX MuJoCo model."""
#     import pytest as _pytest

#     mj_model = handler._mj_model
#     robot_name = handler._robot.name
#     prefix = f"{robot_name}/"

#     for i in range(mj_model.nsite):
#         name = mj_model.site(i).name
#         if name.startswith(prefix):
#             return name

#     _pytest.skip(f"No site with prefix '{prefix}' found in MJX MuJoCo model")


# def site_id_cache_mujoco_query(handler):
#     """Child-process body: validate _get_site_id caching on a real MuJoCo model."""

#     _site_cache.clear()
#     mj_model = handler.physics.model
#     assert mj_model.nsite > 0

#     site_name = mj_model.site(0).name
#     sid1 = _get_site_id(mj_model, site_name)
#     sid2 = _get_site_id(mj_model, site_name)

#     assert sid1 == sid2
#     key = id(mj_model)
#     assert key in _site_cache
#     assert _site_cache[key][site_name] == sid1
#     logger = log.bind(sim="mujoco")
#     logger.info("site-id cache populated correctly for MuJoCo model")


# def site_pos_mujoco_query(handler):
#     """Child-process body: validate SitePos on a real MuJoCo handler."""
#     import torch as _torch

#     full_site_name = _pick_robot_site_name(handler)
#     site_name = full_site_name.split("/", 1)[1]

#     query = SitePos(site_name)
#     query.bind_handler(handler)

#     pos = query()
#     assert isinstance(pos, _torch.Tensor)
#     assert pos.shape == (1, 3)

#     sid = _get_site_id(handler.physics.model, full_site_name)
#     expected = handler.data.site_xpos[sid]
#     assert _torch.allclose(pos.squeeze(0), _torch.as_tensor(expected, dtype=pos.dtype), atol=1e-5)


# def site_pos_mjx_query(handler):
#     """Child-process body: validate SitePos on a real MJX handler."""
#     import torch as _torch

#     full_site_name = _pick_mjx_robot_site_name(handler)
#     site_name = full_site_name.split("/", 1)[1]

#     query = SitePos(site_name)
#     query.bind_handler(handler)

#     pos = query()
#     assert isinstance(pos, _torch.Tensor)
#     assert pos.shape == (handler.num_envs, 3)


# def test_get_site_id_populates_cache_with_real_model(shared_handler):
#     """Use shared_handler; run only when sim == 'mujoco'."""
#     sim, proxy = shared_handler
#     if sim != "mujoco":
#         pytest.skip("Skipping MuJoCo SitePos cache test for non-mujoco sim")

#     pytest.importorskip("mujoco")
#     proxy.run_test(site_id_cache_mujoco_query)


# def test_site_pos_mujoco_returns_world_position_tensor(shared_handler):
#     """SitePos should return a (1, 3) tensor matching MuJoCo's site_xpos."""
#     sim, proxy = shared_handler
#     if sim != "mujoco":
#         pytest.skip("Skipping MuJoCo SitePos test for non-mujoco sim")

#     pytest.importorskip("mujoco")
#     proxy.run_test(site_pos_mujoco_query)


# def test_site_pos_mjx_returns_world_position_tensor(shared_handler):
#     """SitePos should return an (N_env, 3) tensor for MJX."""
#     sim, proxy = shared_handler
#     if sim != "mjx":
#         pytest.skip("Skipping MJX SitePos test for non-mjx sim")

#     pytest.importorskip("mujoco")
#     pytest.importorskip("jax")
#     proxy.run_test(site_pos_mjx_query)
