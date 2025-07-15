import jax
from brax.base import Motion, Transform
from brax.mjx.base import State as MjxState
from mujoco import mjx


def perturbed_pipeline_step(sys, pipeline_state, action, xfrc_applied, n_frames):
    """Takes a physics step using the physics pipeline."""

    def f(state, _):
        data = state.data.replace(ctrl=action, xfrc_applied=xfrc_applied)
        data = mjx.step(sys, data)

        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        contact = None

        return MjxState(q, qd, x, xd, contact, data), None

    return jax.lax.scan(f, pipeline_state, (), n_frames)[0]
