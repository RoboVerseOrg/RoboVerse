import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32
COMPUTE_DTYPE = f32
PARAM_DTYPE = f32
ENABLE_CHECKS = False


def cast_to_compute(values):
    return tree_map(lambda x: x.astype(COMPUTE_DTYPE), values)


def get_param_dtype():
    return PARAM_DTYPE


def check(predicate, message, **kwargs):
    if ENABLE_CHECKS:
        checkify.check(predicate, message, **kwargs)


def parallel():
    try:
        jax.lax.axis_index("i")
        return True
    except NameError:
        return False


def tensorstats(tensor, prefix=None):
    assert tensor.size > 0, tensor.shape
    assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
    tensor = tensor.astype(f32)  # To avoid overflows.
    metrics = {
        "mean": tensor.mean(),
        "std": tensor.std(),
        "mag": jnp.abs(tensor).mean(),
        "min": tensor.min(),
        "max": tensor.max(),
        "dist": subsample(tensor),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def subsample(values, amount=1024):
    values = values.flatten()
    if len(values) > amount:
        values = jax.random.permutation(nj.rng(), values)[:amount]
    return values


def scan(fn, inputs, start, unroll=True, axis=0, modify=False):
    if axis:
        inputs = tree_map(lambda x: x.swapaxes(0, axis), inputs)
    fn2 = lambda carry, inp: (fn(carry, inp),) * 2
    if unroll:
        length = len(jax.tree_util.tree_leaves(inputs)[0])
        carrydef = jax.tree_util.tree_structure(start)
        carry = start
        outs = []
        for index in range(length):
            carry, out = fn2(carry, tree_map(lambda x: x[index], inputs))
            flat, treedef = jax.tree_util.tree_flatten(out)
            assert treedef == carrydef, (treedef, carrydef)
            outs.append(flat)
        outs = [jnp.stack([carry[i] for carry in outs], 0) for i in range(len(outs[0]))]
        outs = carrydef.unflatten(outs)
    else:
        outs = nj.scan(fn2, start, inputs, modify=modify)[1]
    if axis:
        outs = tree_map(lambda x: x.swapaxes(0, axis), outs)
    return outs


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def switch(pred, lhs, rhs):
    def fn(lhs, rhs):
        assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
        mask = pred
        while len(mask.shape) < len(lhs.shape):
            mask = mask[..., None]
        return jnp.where(mask, lhs, rhs)

    return tree_map(fn, lhs, rhs)


def reset(xs, reset):
    def fn(x):
        mask = reset
        while len(mask.shape) < len(x.shape):
            mask = mask[..., None]
        return x * (1 - mask.astype(x.dtype))

    return tree_map(fn, xs)


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=f32):
        super().__init__(logits, probs, dtype)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=(), seed=None):
        sample = sg(super().sample(sample_shape, seed))
        probs = self._pad(super().probs_parameter(), sample.shape)
        return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor


class MSEDist:
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dims, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = jnp.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = jnp.abs(self._mode - symlog(value))
            distance = jnp.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotDist:
    def __init__(self, logits, bins, dims=0, transfwd=None, transbwd=None):
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert logits.dtype == f32, logits.dtype
        assert bins.dtype == f32, bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = jnp.array(bins)
        self.transfwd = transfwd or (lambda x: x)
        self.transbwd = transbwd or (lambda x: x)
        self.batch_shape = logits.shape[: len(logits.shape) - dims - 1]
        self.event_shape = logits.shape[len(logits.shape) - dims : -1]

    def mean(self):
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.transbwd((self.probs * self.bins).sum(-1))
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p1 * b1).sum(-1) + (p2 * b2).sum(-1) + (p3 * b3)[..., ::-1].sum(-1)
            return self.transbwd(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
            wavg = (p1 * b1).sum(-1) + (p2 * b2)[::-1].sum(-1)
            return self.transbwd(wavg)

    def mode(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def log_prob(self, x):
        assert x.dtype == f32, x.dtype
        x = self.transfwd(x)
        below = (self.bins <= x[..., None]).astype(i32).sum(-1) - 1
        above = len(self.bins) - (self.bins > x[..., None]).astype(i32).sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - jax.scipy.special.logsumexp(self.logits, -1, keepdims=True)
        return (target * log_pred).sum(-1).sum(self.dims)


def video_grid(video):
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
    # Values are NaN when there are no positives or negatives in the current
    # batch, which means they will be ignored when aggregating metrics via
    # np.nanmean() later, as they should.
    pos = (target.astype(f32) > thres).astype(f32)
    neg = (target.astype(f32) <= thres).astype(f32)
    pred = (dist.mean().astype(f32) > thres).astype(f32)
    loss = -dist.log_prob(target)
    return dict(
        pos_loss=(loss * pos).sum() / pos.sum(),
        neg_loss=(loss * neg).sum() / neg.sum(),
        pos_acc=(pred * pos).sum() / pos.sum(),
        neg_acc=((1 - pred) * neg).sum() / neg.sum(),
        rate=pos.mean(),
        avg=target.astype(f32).mean(),
        pred=dist.mean().astype(f32).mean(),
    )


class Moments(nj.Module):
    def __init__(self, impl="mean_std", decay=0.99, max=1e8, eps=0.0, perclo=5, perchi=95):
        self.impl = impl
        self.decay = decay
        self.max = max
        self.eps = eps
        self.perclo = perclo
        self.perchi = perchi
        if self.impl == "off":
            pass
        elif self.impl == "mean_std":
            self.step = nj.Variable(jnp.zeros, (), i32, name="step")
            self.mean = nj.Variable(jnp.zeros, (), f32, name="mean")
            self.sqrs = nj.Variable(jnp.zeros, (), f32, name="sqrs")
        elif self.impl == "min_max":
            self.low = nj.Variable(jnp.zeros, (), f32, name="low")
            self.high = nj.Variable(jnp.zeros, (), f32, name="high")
        elif self.impl == "perc_ema":
            self.low = nj.Variable(jnp.zeros, (), f32, name="low")
            self.high = nj.Variable(jnp.zeros, (), f32, name="high")
        elif self.impl == "perc_ema_corr":
            self.step = nj.Variable(jnp.zeros, (), i32, name="step")
            self.low = nj.Variable(jnp.zeros, (), f32, name="low")
            self.high = nj.Variable(jnp.zeros, (), f32, name="high")
        elif self.impl == "mean_mag":
            self.mag = nj.Variable(jnp.zeros, (), f32, name="mag")
        elif self.impl == "max_mag":
            self.mag = nj.Variable(jnp.zeros, (), f32, name="mag")
        else:
            raise NotImplementedError(self.impl)

    def __call__(self, x):
        self.update(x)
        return self.stats()

    def update(self, x):
        if parallel():
            mean = lambda x: jax.lax.pmean(x.mean(), "i")
            min_ = lambda x: jax.lax.pmin(x.min(), "i")
            max_ = lambda x: jax.lax.pmax(x.max(), "i")
            per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, "i"), q)
        else:
            mean = jnp.mean
            min_ = jnp.min
            max_ = jnp.max
            per = jnp.percentile
        x = sg(x.astype(f32))
        m = self.decay
        if self.impl == "off":
            pass
        elif self.impl == "mean_std":
            self.step.write(self.step.read() + 1)
            self.mean.write(m * self.mean.read() + (1 - m) * mean(x))
            self.sqrs.write(m * self.sqrs.read() + (1 - m) * mean(x * x))
        elif self.impl == "min_max":
            low, high = min_(x), max_(x)
            self.low.write(m * jnp.minimum(self.low.read(), low) + (1 - m) * low)
            self.high.write(m * jnp.maximum(self.high.read(), high) + (1 - m) * high)
        elif self.impl == "perc_ema":
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.write(m * self.low.read() + (1 - m) * low)
            self.high.write(m * self.high.read() + (1 - m) * high)
        elif self.impl == "perc_ema_corr":
            self.step.write(self.step.read() + 1)
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.write(m * self.low.read() + (1 - m) * low)
            self.high.write(m * self.high.read() + (1 - m) * high)
        elif self.impl == "mean_mag":
            curr = mean(jnp.abs(x))
            self.mag.write(m * self.mag.read() + (1 - m) * curr)
        elif self.impl == "max_mag":
            curr = max_(jnp.abs(x))
            self.mag.write(m * jnp.maximum(self.mag.read(), curr) + (1 - m) * curr)
        else:
            raise NotImplementedError(self.impl)

    def stats(self):
        if self.impl == "off":
            return 0.0, 1.0
        elif self.impl == "mean_std":
            corr = 1 - self.decay ** self.step.read().astype(f32)
            mean = self.mean.read() / corr
            var = (self.sqrs.read() / corr) - self.mean.read() ** 2
            std = jnp.sqrt(jnp.maximum(var, 1 / self.max**2) + self.eps)
            return sg(mean), sg(std)
        elif self.impl == "min_max":
            offset = self.low.read()
            invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
            return sg(offset), sg(invscale)
        elif self.impl == "perc_ema":
            offset = self.low.read()
            invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
            return sg(offset), sg(invscale)
        elif self.impl == "perc_ema_corr":
            corr = 1 - self.decay ** self.step.read().astype(f32)
            lo = self.low.read() / corr
            hi = self.high.read() / corr
            invscale = jnp.maximum(1 / self.max, hi - lo)
            return sg(lo), sg(invscale)
        elif self.impl == "mean_mag":
            offset = jnp.array(0)
            invscale = jnp.maximum(1 / self.max, self.mag.read())
            return sg(offset), sg(invscale)
        elif self.impl == "max_mag":
            offset = jnp.array(0)
            invscale = jnp.maximum(1 / self.max, self.mag.read())
            return sg(offset), sg(invscale)
        else:
            raise NotImplementedError(self.impl)


class Optimizer(nj.Module):
    def __init__(
        self,
        lr,
        opt="adam",
        eps=1e-5,
        clip=100.0,
        warmup=0,
        wd=0.0,
        wd_pattern=r"/(w|kernel)$",
        adaclip=0.0,
        init_grad_scale=1e4,
    ):
        assert wd_pattern[0] not in ("0", "1")
        wd_pattern = re.compile(wd_pattern)
        chain = []
        if clip:
            chain.append(optax.clip_by_global_norm(clip))
        if opt == "adam":
            chain.append(optax.scale_by_adam(eps=eps))
        elif opt == "lion":
            chain.append(optax.scale_by_lion())
        else:
            raise NotImplementedError(opt)
        if adaclip:
            chain.append(ada_clip(adaclip))
        if wd:
            chain.append(
                optax.additive_weight_decay(
                    wd,
                    lambda params: (tree_map(lambda k: bool(wd_pattern.search(k)), tree_keys(params))),
                )
            )
        chain.append(optax.scale(-lr))
        self.opt = optax.chain(*chain)
        self.warmup = warmup
        self.step = nj.Variable(jnp.array, 0, i32, name="step")
        self.scaling = COMPUTE_DTYPE == jnp.float16
        if self.scaling:
            self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
            self.grad_scale = nj.Variable(jnp.array, init_grad_scale, f32, name="grad_scale")
            self.good_steps = nj.Variable(jnp.array, 0, i32, name="good_steps")
        self.once = True

    def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
        def wrapped(*args, **kwargs):
            outs = lossfn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        metrics = {}
        loss, params, grads, aux = nj.grad(wrapped, modules, has_aux=True)(*args, **kwargs)
        if not isinstance(modules, (list, tuple)):
            modules = [modules]

        param_count = sum([np.prod(x.shape) for x in params.values()])
        if self.once:
            self.once = False
            print(f"Optimizer {self.name} has {param_count:,} variables.")

        if parallel():
            grads = tree_map(lambda x: jax.lax.pmean(x, "i"), grads)
        if self.scaling:
            invscale = 1.0 / self.grad_scale.read()
            grads = tree_map(lambda x: x * invscale, grads)
        optstate = self.get("state", self.opt.init, params)
        updates, optstate = self.opt.update(grads, optstate, params)
        self.put("state", optstate)
        if self.warmup > 0:
            scale = jnp.clip(self.step.read().astype(f32) / self.warmup, 0, 1)
            updates = tree_map(lambda x: x * scale, updates)
        nj.context().update(optax.apply_updates(params, updates))
        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)
        param_norm = optax.global_norm([x.getm() for x in modules])
        isfin = jnp.isfinite
        if self.scaling:
            self._update_scale(grads, jnp.isfinite(grad_norm))
            metrics["grad_scale"] = self.grad_scale.read()
            metrics["grad_overflow"] = (~jnp.isfinite(grad_norm)).astype(f32)
            grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
            self.step.write(self.step.read() + isfin(grad_norm).astype(i32))
        else:
            check(isfin(grad_norm), f"{self.path} grad norm: {{x}}", x=grad_norm)
            self.step.write(self.step.read() + 1)
        check(isfin(update_norm), f"{self.path} updates: {{x}}", x=update_norm)
        check(isfin(param_norm), f"{self.path} params: {{x}}", x=param_norm)

        metrics["loss"] = loss.mean()
        metrics["grad_norm"] = grad_norm
        metrics["update_norm"] = update_norm
        metrics["param_norm"] = param_norm
        metrics["grad_steps"] = self.step.read()
        metrics["param_count"] = param_count
        metrics = {f"{self.name}_{k}": v for k, v in metrics.items()}
        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, grads, finite):
        keep = finite & (self.good_steps.read() < 1000)
        incr = finite & (self.good_steps.read() >= 1000)
        decr = ~finite
        self.good_steps.write(keep.astype(i32) * (self.good_steps.read() + 1))
        self.grad_scale.write(
            jnp.clip(
                keep.astype(f32) * self.grad_scale.read()
                + incr.astype(f32) * self.grad_scale.read() * 2
                + decr.astype(f32) * self.grad_scale.read() / 2,
                1e-4,
                1e4,
            )
        )
        return finite


def ada_clip(clip=0.1, b1=0.9, b2=0.999, eps=1e-6):
    def init_fn(params):
        m = jnp.zeros((), f32)
        v = jnp.zeros((), f32)
        s = jnp.ones((), i32)
        return m, v, s

    def update_fn(updates, state, params):
        m, v, s = state
        x = optax.global_norm(updates)
        m_corr = m / (1 - b1 ** s.astype(f32))
        v_corr = v / (1 - b2 ** s.astype(f32))
        up = m_corr + clip * jnp.sqrt(v_corr + eps)
        up = jax.lax.select(s <= 1, jnp.inf, up)
        trigger = jnp.squeeze(x < up)
        updates = jax.tree_util.tree_map(lambda t: jax.lax.select(trigger, t, (t / x.astype(t.dtype)) * up), updates)
        x = jnp.minimum(x, up)
        m = b1 * m + (1 - b1) * x
        v = b2 * v + (1 - b2) * x * x
        s = s + 1
        return updates, (m, v, s)

    return optax.GradientTransformation(init_fn, update_fn)


# def late_grad_clip(value=1.0):
#   def init_fn(params):
#     return ()
#   def update_fn(updates, state, params):
#     updates = tree_map(lambda x: jnp.clip(x, -value, value), updates)
#     return updates, ()
#   return optax.GradientTransformation(init_fn, update_fn)


def concat_dict(mapping, batch_shape=None):
    tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
    if batch_shape is not None:
        tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
    return jnp.concatenate(tensors, -1)


def tree_keys(params, prefix=""):
    if hasattr(params, "items"):
        return type(params)({k: tree_keys(v, prefix + "/" + k.lstrip("/")) for k, v in params.items()})
    elif isinstance(params, (tuple, list)):
        return [tree_keys(x, prefix) for x in params]
    elif isinstance(params, jnp.ndarray):
        return prefix
    else:
        raise TypeError(type(params))


class SlowUpdater(nj.Module):
    def __init__(self, src, dst, fraction=1.0, period=1):
        self.src = src
        self.dst = dst
        self.fraction = fraction
        self.period = period
        self.updates = nj.Variable(jnp.zeros, (), i32, name="updates")

    def __call__(self):
        assert self.src.getm()
        updates = self.updates.read()
        need_init = (updates == 0).astype(f32)
        need_update = (updates % self.period == 0).astype(f32)
        mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
        params = {k.replace(f"/{self.src.name}/", f"/{self.dst.name}/"): v for k, v in self.src.getm().items()}
        ema = tree_map(lambda s, d: mix * s + (1 - mix) * d, params, self.dst.getm())
        for name, param in ema.items():
            assert param.dtype == jnp.float32, f"EMA of {name} should be float32 not {param.dtype}"
        self.dst.putm(ema)
        self.updates.write(updates + 1)
