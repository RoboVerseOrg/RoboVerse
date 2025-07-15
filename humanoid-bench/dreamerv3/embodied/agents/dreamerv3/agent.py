import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
f32 = jnp.float32
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
is_list = lambda x: isinstance(x, list)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj

# from . import ssm


@jaxagent.Wrapper
class Agent(nj.Module):
    configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "configs.yaml").read())

    def __init__(self, obs_space, act_space, config):
        self.obs_space = {k: v for k, v in obs_space.items() if not k.startswith("log_")}
        self.act_space = {k: v for k, v in act_space.items() if k != "reset"}
        self.config = config
        self.wm = WorldModel(self.obs_space, self.act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name="task_behavior"
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name="expl_behavior"
            )

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, carry, mode="train"):
        carry = tree_map(
            lambda x: jnp.stack(x, 0) if isinstance(x, list) else x,
            carry,
            is_leaf=is_list,
        )
        self.config.jax.jit and embodied.print("Tracing policy function", "yellow")
        obs = self.preprocess(obs)
        (prev_state, prev_action), task_state, expl_state = carry
        embed = self.wm.encoder(obs, batchdims=1)
        carry = self.wm.rssm.obs_step(prev_state, prev_action, embed, obs["is_first"])
        task_act, task_state = self.task_behavior.policy(carry, task_state)
        expl_act, expl_state = self.expl_behavior.policy(carry, expl_state)
        act = {"eval": task_act, "explore": expl_act, "train": task_act}[mode]
        carry = ((carry, act), task_state, expl_state)
        act = {k: jnp.argmax(act[k], -1).astype(jnp.int32) if s.discrete else act[k] for k, s in self.act_space.items()}
        return act, carry

    def train(self, data, carry, train_wm=True, train_ac=True, ignore_inputs=()):
        self.config.jax.jit and embodied.print("Tracing train function", "yellow")
        data = self.preprocess(data)
        for key in ignore_inputs:
            data[key] = jnp.zeros_like(data[key])
        metrics = {}
        if train_wm:
            outs, carry, mets = self.wm.train(data, carry, ignore_inputs)
            metrics.update(mets)
        else:
            _, outs, carry = self.wm.observe(data, carry)
        if train_ac:
            context = {**data, **outs}
            start = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), context)
            _, mets = self.task_behavior.train(self.wm.imagine, start, context)
            metrics.update(mets)
            if self.config.expl_behavior != "None":
                _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
                metrics.update({"expl_" + key: value for key, value in mets.items()})
        outs = {}
        return outs, carry, metrics

    def report(self, data):
        self.config.jax.jit and embodied.print("Tracing report function", "yellow")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        spaces = {**self.obs_space, **self.act_space}
        result = {}
        for key, value in obs.items():
            if key.startswith("log_") or key in ("reset", "key", "id"):
                continue
            space = spaces[key]
            if key in self.act_space and space.discrete:
                value = jax.nn.one_hot(value, int(space.high))
            if len(space.shape) >= 3 and space.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            result[key] = value
        result["cont"] = 1.0 - result["is_terminal"].astype(f32)
        return result


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.encoder = nets.MultiEncoder(obs_space, **config.encoder, name="enc")
        if config.rssm_type == "rssm":
            self.rssm = nets.RSSM(**config.rssm, name="rssm")
        else:
            raise NotImplementedError(config.rssm_type)
        self.heads = {
            "decoder": nets.MultiDecoder(obs_space, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }

        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        cnn = scales.pop("dec_cnn")
        mlp = scales.pop("dec_mlp")
        scales.update({k: cnn for k in self.heads["decoder"].cnn_keys})
        scales.update({k: mlp for k in self.heads["decoder"].mlp_keys})
        self.scales = scales

    def initial(self, batch_size):
        bs = batch_size
        latent = self.rssm.initial(bs)
        action = {
            k: jnp.zeros((bs, *v.shape, int(v.high)) if v.discrete else (bs, *v.shape))
            for k, v in self.act_space.items()
        }
        return latent, action

    def train(self, data, carry, ignore_inputs=()):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (outs, carry, metrics) = self.opt(modules, self.loss, data, carry, ignore_inputs, has_aux=True)
        metrics.update(mets)
        return outs, carry, metrics

    def loss(self, data, carry, ignore_inputs=()):
        metrics = {}
        states, feats, carry = self.observe(data, carry)

        dists = {}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        dists = {k: v for k, v in dists.items() if k not in ignore_inputs}
        losses, stats = self.rssm.loss(states, **self.config.rssm_loss)

        for key, dist in dists.items():
            try:
                loss = -dist.log_prob(data[key].astype(f32))
            except Exception as e:
                raise Exception(f"Error in {name} loss:\n\n{e}.") from e
            assert loss.shape == feats["embed"].shape[:2], (key, loss.shape)
            losses[key] = loss

        for key, dist in dists.items():
            if hasattr(dist, "entropy"):
                metrics[f"{key}_head_entropy"] = dist.entropy().mean()
            if isinstance(dist, tfd.Categorical):
                accuracy = (dist.mode() == data[key]).astype(f32)
                metrics[f"{key}_head_accuracy"] = accuracy.mean()

        scaled = {k: v.mean() * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        assert model_loss.shape == ()
        out.update({f"{k}_loss": v for k, v in losses.items()})
        metrics.update(self._metrics(data, dists, states, stats, losses, model_loss))
        return model_loss, (feats, carry, metrics)

    def observe(self, data, carry):
        embed = self.encoder(data)
        prev_state, prev_action = carry
        prev_acts = {k: jnp.concatenate([prev_action[k][:, None], data[k][:, :-1]], 1) for k in self.act_space}
        states = self.rssm.observe(prev_state, prev_acts, embed, data["is_first"])
        new_state = {k: v[:, -1] for k, v in states.items()}
        new_action = {k: data[k][:, -1] for k in self.act_space}
        carry = (new_state, new_action)
        outs = {**states, "embed": embed}
        return states, outs, carry

    def imagine(self, policy, start, horizon, carry=None):
        carry = carry or {}
        state_keys = list(self.rssm.initial(1).keys())
        state = {k: v for k, v in start.items() if k in state_keys}
        action, carry = policy(state, carry)
        keys = list(state.keys()) + list(action.keys()) + list(carry.keys())
        assert len(set(keys)) == len(keys), ("Colliding keys", keys)

        def step(prev, _):
            state, action, carry = prev
            state = self.rssm.img_step(state, action)
            action, carry = policy(state, carry)
            return state, action, carry

        states, actions, carries = jaxutils.scan(
            step, jnp.arange(horizon), (state, action, carry), self.config.imag_unroll
        )
        states, actions, carries = tree_map(
            lambda traj, first: jnp.concatenate([first[None], traj], 0),
            (states, actions, carries),
            (state, action, carry),
        )
        traj = {**states, **actions, **carries}
        if self.config.imag_cont == "mode":
            cont = self.heads["cont"](traj).mode()
        elif self.config.imag_cont == "mean":
            cont = self.heads["cont"](traj).mean()
        else:
            raise NotImplementedError(self.config.imag_cont)
        first_cont = (1.0 - start["is_terminal"]).astype(f32)
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        states, _, _ = self.observe(
            {k: v[:6, :5] for k, v in data.items()},
            self.initial(len(data["is_first"][:6])),
        )
        start = {k: v[:, -1] for k, v in states.items()}
        recon = self.heads["decoder"](states)
        prev_acts = {k: data[k][:6, 5 - 1 : -1] for k in self.act_space}
        openl = self.heads["decoder"](self.rssm.imagine(start, prev_acts))
        for key in self.heads["decoder"].cnn_keys:
            truth = data[key][:6].astype(f32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, states, stats, losses, model_loss):
        metrics = {}
        metrics.update(jaxutils.tensorstats(stats["prior_ent"], "prior_ent"))
        metrics.update(jaxutils.tensorstats(stats["post_ent"], "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss"] = model_loss
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists:  # and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists:  # and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config, act_priors=None):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.act_priors = act_priors or {}
        self.config = config
        if len(act_space) == 1 and list(act_space.values())[0].discrete:
            self.grad = config.actor_grad_disc
        elif len(act_space) == 1:
            self.grad = config.actor_grad_cont
        else:
            self.grad = "reinforce"
        dist1, dist2 = config.actor_dist_disc, config.actor_dist_cont
        shapes = {k: (*s.shape, int(s.high)) if s.discrete else s.shape for k, s in act_space.items()}
        dists = {k: dist1 if v.discrete else dist2 for k, v in act_space.items()}
        self.actor = nets.MLP(**config.actor, name="actor", shape=shapes, dist=dists)
        self.retnorms = {k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry, sample=True):
        dist = self.actor(sg(state), batchdims=1)
        if sample:
            action = {k: v.sample(seed=nj.rng()) for k, v in dist.items()}
        else:
            action = {k: v.mode() for k, v in dist.items()}
        return action, carry

    def train(self, imagine, start, context):
        def loss(start):
            traj = imagine(self.policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = {k: v.log_prob(sg(traj[k]))[:-1] for k, v in policy.items()}
        loss = {
            "backprop": -adv,
            "reinforce": -sum(logpi.values()) * sg(adv),
        }[self.grad]
        ent = {k: v.entropy()[:-1] for k, v in policy.items()}
        for key, fn in self.act_priors.items():
            pi = policy[key]
            if isinstance(pi, jaxutils.OneHotDist):
                pi = tfd.Categorical(pi.logits_parameter())
            ent[key] = -pi.kl_divergence(fn(sg(traj)))[:-1]
        loss -= self.config.actent * sum(ent.values())
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        for key, space in self.act_space.items():
            act = jnp.argmax(traj[key], -1) if space.discrete else traj[key]
            metrics.update(jaxutils.tensorstats(act.astype(f32), f"{key}_action"))
            rand = (ent[key] - policy[key].minent) / (policy[key].maxent - policy[key].minent)
            rand = rand.mean(range(2, len(rand.shape)))
            metrics.update(jaxutils.tensorstats(rand, f"{key}_policy_randomness"))
            metrics.update(jaxutils.tensorstats(ent[key], f"{key}_policy_entropy"))
            metrics.update(jaxutils.tensorstats(logpi[key], f"{key}_policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", **self.config.critic)
        self.slow = nets.MLP((), name="slow", **self.config.critic, dtype="float32")
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
            name="updater",
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj, slow=self.config.slow_critic_target)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None, slow=False):
        rew = self.rewfn(traj)
        assert len(rew) == len(list(traj.values())[0]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        if slow:
            value = self.slow(traj).mean()
        else:
            value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
