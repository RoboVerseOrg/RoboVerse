"""Integration tests for metasim/queries/site.py using real MuJoCo and MJX."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.queries.site import SitePos, _get_site_id, _site_cache


def _pick_robot_site_name(handler, is_mjx: bool = False) -> str:
    """Pick a site name belonging to the robot from the MuJoCo/MJX model.

    Args:
        handler: The simulator handler (MuJoCo or MJX).
        is_mjx: Whether this is an MJX handler (uses different attribute names).
    """
    import pytest as _pytest

    if is_mjx:
        mj_model = handler._mj_model
        # Use the MJCF identifier which matches site name prefixes
        prefix = handler._mujoco_robot_name
    else:
        mj_model = handler.physics.model
        # Use the MJCF identifier which matches site name prefixes
        prefix = handler._mujoco_robot_names[0]

    for i in range(mj_model.nsite):
        name = mj_model.site(i).name
        if name.startswith(prefix):
            return name

    _pytest.skip(f"No site with prefix '{prefix}' found in {'MJX' if is_mjx else 'MuJoCo'} model")


@pytest.mark.mujoco
def test_get_site_id_populates_cache_with_real_model(handler):
    """Use shared_handler; run only when sim == 'mujoco'."""
    _site_cache.clear()
    mj_model = handler.physics.model
    assert mj_model.nsite > 0

    site_name = mj_model.site(0).name
    sid1 = _get_site_id(mj_model, site_name)
    sid2 = _get_site_id(mj_model, site_name)

    assert sid1 == sid2
    key = id(mj_model)
    assert key in _site_cache
    assert _site_cache[key][site_name] == sid1
    logger = log.bind(sim="mujoco")
    logger.info("site-id cache populated correctly for MuJoCo model")


@pytest.mark.mujoco
def test_site_pos_mujoco_returns_world_position_tensor(handler):
    """SitePos should return a (1, 3) tensor matching MuJoCo's site_xpos."""
    import torch as _torch

    full_site_name = _pick_robot_site_name(handler)
    site_name = full_site_name.split("/", 1)[1]

    query = SitePos(site_name)
    query.bind_handler(handler)

    pos = query()
    assert isinstance(pos, _torch.Tensor)
    assert pos.shape == (1, 3)

    sid = _get_site_id(handler.physics.model, full_site_name)
    expected = handler.data.site_xpos[sid]
    assert _torch.allclose(pos.squeeze(0), _torch.as_tensor(expected, dtype=pos.dtype), atol=1e-5)


@pytest.mark.mjx
def test_site_pos_mjx_returns_world_position_tensor(handler):
    """SitePos should return an (N_env, 3) tensor for MJX."""
    import torch as _torch

    full_site_name = _pick_robot_site_name(handler, is_mjx=True)
    site_name = full_site_name.split("/", 1)[1]

    query = SitePos(site_name)
    query.bind_handler(handler)

    pos = query()
    assert isinstance(pos, _torch.Tensor)
    assert pos.shape == (handler.num_envs, 3)
