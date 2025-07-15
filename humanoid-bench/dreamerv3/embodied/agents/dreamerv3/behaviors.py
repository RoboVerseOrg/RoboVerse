import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from . import agent, expl, jaxutils
from . import ninjax as nj


class Greedy(nj.Module):
    def __init__(self, wm, act_space, config):
        rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        rewfn = lambda s, f=rewfn: f(s) * config.reward_scales.extr
        if config.critic_type == "vfunction":
            critics = {"extr": agent.VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)
        act_priors = None
        if config.talk_prior:
            act_priors = {"talk": lambda s: wm.heads["decoder"](s, cnn=False)["text"]}
        self.ac = agent.ImagActorCritic(critics, {"extr": 1.0}, act_space, config, act_priors, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(nj.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.dists = {k: self._make_dist(s) for k, s in act_space.items()}

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, latent, state):
        action = {k: v.sample(len(state), seed=nj.rng()) for k, v in self.dists.items()}
        return action, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}

    def _make_dist(self, space):
        if space.discrete:
            dist = jaxutils.OneHotDist(jnp.zeros((*space.shape, space.high)))
        else:
            dist = tfd.Independent(tfd.Uniform(space.low, space.high), 1)
        return dist


class Explore(nj.Module):
    REWARDS = {
        "disag": expl.Disag,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == "extr":
                rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
                rewfn = lambda s, f=rewfn: f(s) * config.reward_scales.extr
                critics[key] = agent.VFunction(rewfn, config, name=key)
            else:
                rewfn = self.REWARDS[key](wm, act_space, config, name=key + "_reward")
                rewfn = lambda s, f=rewfn: f(s) * config.reward_scales[key]
                critics[key] = agent.VFunction(rewfn, config, name=key)
                self.rewards[key] = rewfn
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent.ImagActorCritic(critics, scales, act_space, config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        metrics = {}
        for key, rewfn in self.rewards.items():
            mets = rewfn.train(data)
            metrics.update({f"{key}_k": v for k, v in mets.items()})
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}
