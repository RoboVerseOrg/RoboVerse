import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn.functional as F

import math

np.set_printoptions(precision=30, floatmode="maxprec")
prec = 10

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def equal(a0, a1):
    return np.array_equal(np.round(a0, prec), np.round(a1, prec))


def safe_acos(q):
    """
    pytorch acos nan: https://github.com/pytorch/pytorch/issues/8069
    """
    return torch.acos(torch.clamp(q, -1.0 + 1e-7, 1.0 - 1e-7))


def euler_from_quaternion():
    pass


def quaternion_from_euler(ai, aj, ak, axes="sxyz"):
    """ "
    Input: ai, aj, ak: Bx1
    Output: quat: Bx4
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    B = ai.size()[0]
    ai, aj, ak = ai.clone(), aj.clone(), ak.clone()

    device = ai.device
    dtype = ai.dtype

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak.clone(), ai.clone()
    if parity:
        aj = -aj.clone()

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = torch.cos(ai)
    si = torch.sin(ai)
    cj = torch.cos(aj)
    sj = torch.sin(aj)
    ck = torch.cos(ak)
    sk = torch.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = torch.tensor([1.0, 0.0, 0.0, 0.0] * B, dtype=dtype, device=device).view(B, 4)
    if repetition:
        q[:, 0] = cj * (cc - ss)
        q[:, i] = cj * (cs + sc)
        q[:, j] = sj * (cc + ss)
        q[:, k] = sj * (cs - sc)
    else:
        q[:, 0] = cj * cc + sj * ss
        q[:, i] = cj * sc - sj * cs
        q[:, j] = cj * ss + sj * cc
        q[:, k] = cj * cs - sj * sc
    if parity:
        q[:, j] *= -1.0

    return q


def get_angvel_fd_batch(prev_bquat, cur_bquat, dt):
    q_diff = quaternion_multiply_batch(
        cur_bquat.reshape(-1, 4), quaternion_inverse_batch(prev_bquat.reshape(-1, 4))
    )
    body_angvel = rotation_from_quaternion_batch(q_diff) / dt
    return body_angvel.reshape(prev_bquat.shape[0], prev_bquat.shape[1], 3)


def rotation_from_quaternion(_q, separate=False):
    q = _q.clone()
    device = q.device
    dtype = q.dtype
    if 1 - q[0] < 1e-6:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        angle = 0.0
    else:
        axis = q[1:4] / torch.sqrt(torch.abs(1 - q[0] * q[0]))
        angle = 2 * safe_acos(q[0])
    return (axis, angle) if separate else axis * angle


def rotation_from_quaternion_batch(_q, separate=False):
    """
    q: size(Bx4)
    Output: size(Bx3)
    """

    assert _q.shape[-1] == 4

    q = _q.clone()
    B = q.size()[0]
    device = q.device
    dtype = q.dtype
    zero_axis = torch.tensor([1.0, 0.0, 0.0] * B, dtype=dtype, device=device).view(B, 3)
    zero_angle = torch.tensor([0.0] * B, dtype=dtype, device=device)

    # q = F.normalize(q,p=2, dim=1)

    cond = torch.abs(torch.sin(safe_acos(q[:, 0]))) < 1e-5
    axis = torch.where(
        cond.unsqueeze(1).repeat(1, 3),
        zero_axis,
        q[:, 1:4] / (torch.sin(safe_acos(q[:, 0]))).view(B, 1),
    )
    angle = torch.where(cond, zero_angle, 2 * safe_acos(q[:, 0]))
    assert angle.size()[0] == axis.size()[0]
    return (axis, angle) if separate else axis * angle.view(B, 1)


def quaternion_matrix(_q):
    q = _q.clone()
    n = torch.dot(q, q)
    dtype = q.dtype

    device = q.device
    if n < _EPS:
        return torch.eye(4, dtype=dtype, device=device)
    q = q * torch.sqrt(2.0 / n)
    q = torch.ger(q, q)
    return torch.tensor(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ],
        dtype=dtype,
        device=device,
    )


def quaternion_matrix_batch(_q):
    # ZL: from YE, needs to be changed
    q = _q.clone()
    q_norm = torch.norm(q, dim=1).view(-1, 1)
    q = q / q_norm
    tx = q[..., 1] * 2.0
    ty = q[..., 2] * 2.0
    tz = q[..., 3] * 2.0
    twx = tx * q[..., 0]
    twy = ty * q[..., 0]
    twz = tz * q[..., 0]
    txx = tx * q[..., 1]
    txy = ty * q[..., 1]
    txz = tz * q[..., 1]
    tyy = ty * q[..., 2]
    tyz = tz * q[..., 2]
    tzz = tz * q[..., 3]
    res = torch.stack(
        (
            torch.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=1),
            torch.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=1),
            torch.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=1),
        ),
        dim=2,
    )
    #     res = torch.zeros(res.shape).to(_q.device)
    return res


def quaternion_about_axis(angle, axis):
    device = angle.device
    dtype = angle.dtype
    q = torch.tensor([0.0, axis[0], axis[1], axis[2]], dtype=dtype, device=device)
    qlen = torch.norm(q, p=2)
    if qlen > _EPS:
        q = q * torch.sin(angle / 2.0) / qlen
    q[0] = torch.cos(angle / 2.0)

    return q


def quat_from_expmap(e):
    device = e.device
    dtype = e.dtype
    angle = torch.norm(e, p=2)

    if angle < 1e-8:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        axis = e / angle

    return quaternion_about_axis(angle, axis)


def quaternion_about_axis(angle, axis):
    device = angle.device
    dtype = angle.dtype
    q = torch.tensor([0.0, axis[0], axis[1], axis[2]], dtype=dtype, device=device)
    qlen = torch.norm(q, p=2)
    if qlen > _EPS:
        q = q * torch.sin(angle / 2.0) / qlen
    q[0] = torch.cos(angle / 2.0)

    return q


def quat_from_expmap(e):
    device = e.device
    dtype = e.dtype
    angle = torch.norm(e, p=2)

    if angle < 1e-8:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    else:
        axis = e / angle

    return quaternion_about_axis(angle, axis)


def quaternion_about_axis_batch(angle, axis):
    device = angle.device
    dtype = angle.dtype
    batch_size, _ = angle.shape
    q = torch.zeros((batch_size, 4)).to(device).type(dtype)
    q[:, 1] = axis[:, 0]
    q[:, 2] = axis[:, 1]
    q[:, 3] = axis[:, 2]

    qlen = torch.norm(q, dim=1, p=2)
    q_change = (
        q[qlen > _EPS, :]
        * torch.sin(angle[qlen > _EPS, :] / 2.0)
        / qlen[qlen > _EPS].view(-1, 1)
    )
    q_res = q.clone()
    q_res[qlen > _EPS, :] = q_change
    q_res[:, 0:1] = torch.cos(angle / 2.0)
    return q_res


def quat_from_expmap_batch(e):
    device = e.device
    dtype = e.dtype
    angle = torch.norm(e, dim=1, p=2)
    axis = torch.zeros(e.shape).to(device).type(dtype)

    axis[angle < 1e-8] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    axis[angle >= 1e-8] = e[angle >= 1e-8] / angle[angle >= 1e-8].view(-1, 1)
    return quaternion_about_axis_batch(angle.view(-1, 1), axis)


def quaternion_inverse(_q):
    q = _q.clone()
    q[1:] = -1 * q[1:]
    return q / torch.dot(q, q)


def quaternion_inverse_batch(_q):
    """q: size(Bx4)
    Output: size(Bx4)
    """
    q = _q.clone()
    q_chnage = -1.0 * q[:, 1:]
    q[:, 1:] = q_chnage
    out = q / torch.einsum("bs,bs->b", q, q).unsqueeze(1).repeat(1, 4)
    return out


def transform_vec(v, q, trans="root"):
    device = q.device
    if trans == "root":
        rot = quaternion_matrix(q)[:3, :3]
    elif trans == "heading":
        hq = q.clone()
        hq[1] = 0.0
        hq[2] = 0.0
        hq = hq / torch.norm(hq, p=2)
        rot = quaternion_matrix(hq)[:3, :3]
    else:
        assert False

    v = torch.matmul(torch.transpose(rot, 0, 1), v)


def transform_vec_batch(v, q, trans="root"):
    device = q.device
    if trans == "root":
        rot = quaternion_matrix_batch(q)
    elif trans == "heading":
        hq = get_heading_q_batch(q)
        rot = quaternion_matrix_batch(hq)
    else:
        assert False
    v = torch.matmul(torch.transpose(rot, 1, 2), v.unsqueeze(2))
    return v.squeeze(2)


def get_qvel_fd(cur_qpos, next_qpos, dt, transform=None):
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(cur_qpos[3:7]))
    axis, angle = rotation_from_quaternion(qrel, True)
    if angle > np.pi:  # -180 < angle < 180
        angle -= 2 * np.pi  #
    elif angle < -np.pi:
        angle += 2 * np.pi
    rv = (axis * angle) / dt

    rv = transform_vec(rv, cur_qpos[3:7], "root")
    qvel = (next_qpos[7:] - cur_qpos[7:]) / dt
    qvel = torch.cat((v, rv, qvel))
    if transform is not None:
        v = transform_vec(v, cur_qpos[:, 3:7], transform)
        qvel[:, :3] = v

    return qvel


def get_qvel_fd_batch(cur_qpos, next_qpos, dt, transform=None):
    v = (next_qpos[:, :3] - cur_qpos[:, :3]) / dt
    qrel = quaternion_multiply_batch(
        next_qpos[:, 3:7], quaternion_inverse_batch(cur_qpos[:, 3:7])
    )
    axis, angle = rotation_from_quaternion_batch(qrel, True)

    angle[angle > np.pi] -= 2 * np.pi
    angle[angle < -np.pi] += 2 * np.pi

    rv = (axis * angle.view(-1, 1)) / dt

    rv = transform_vec_batch(rv, cur_qpos[:, 3:7], "root")
    qvel = (next_qpos[:, 7:] - cur_qpos[:, 7:]) / dt
    qvel = torch.cat((v, rv, qvel), dim=1)
    if transform is not None:
        v = transform_vec(v, cur_qpos[3:7], transform)
        qvel[:3] = v
    return qvel


def quaternion_multiply(_q1, _q0):
    q0 = _q0.clone()
    q1 = _q1.clone()
    device = q0.device
    dtype = q0.dtype
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return torch.tensor(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ],
        dtype=dtype,
        device=device,
    )


def quaternion_multiply_batch(q0, q1):
    """
    Multiply quaternion(s) q0 with quaternion(s) q1.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q1.shape[-1] == 4
    assert q0.shape[-1] == 4

    original_shape = q0.shape

    # Compute outer product
    q1_view = q1.view(-1, 4, 1).clone()
    q0_view = q0.view(-1, 1, 4).clone()
    terms = torch.bmm(q1_view, q0_view)

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def get_heading_q(_q):
    q = _q.clone()
    q[1] = 0.0
    q[2] = 0.0
    q_norm = torch.norm(q, p=2)
    return q / q_norm


def get_heading_q_batch(_q):
    q = _q.clone()
    q[:, 1] = 0.0
    q[:, 2] = 0.0

    q_norm = torch.norm(q, dim=1, p=2).view(-1, 1)
    return q / q_norm


def de_heading_batch(q):
    q_deheaded = get_heading_q_batch(q)
    q_deheaded_inv = quaternion_inverse_batch(q_deheaded)

    return quaternion_multiply_batch(q_deheaded_inv, q)
    
def get_heading_batch(q):
    hq = q.clone()
    hq[:, 1] = 0
    hq[:, 2] = 0
    indices = hq[:, 3] < 0.0
    new_vals = -1 * hq[indices, :].clone()
    hq[indices, :] = new_vals
    hq = hq / torch.norm(hq, p=2, dim=1)[:, None]
    w = 2 * safe_acos_batch(hq[:, 0])
    heading = torch.tensor(w, dtype=hq.dtype, device=hq.device)[:, None]
    return heading

def get_heading(q):
    hq = q.clone()
    hq[1] = 0
    hq[2] = 0
    if hq[3] < 0.0:
        hq = -1.0 * hq
    hq = hq / torch.norm(hq, p=2)
    w = 2 * safe_acos(hq[0])
    heading = torch.tensor([w], dtype=hq.dtype, device=hq.device)
    return heading


def safe_acos_batch(q):
    """
    pytorch acos nan: https://github.com/pytorch/pytorch/issues/8069
    """
    return torch.acos(torch.clamp(q, -1.0 + 1e-7, 1.0 - 1e-7))


def quat_mul_vec(q, v):
    return torch.matmul(quaternion_matrix(q)[:3, :3], v)


def quat_mul_vec_batch(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
