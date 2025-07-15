"""Implementations of algorithms for continuous control."""

import functools

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, nonpytree_field, target_update
from jaxrl_m.networks import Critic, Policy, ensemblize
from jaxrl_m.typing import *


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param("log_temp", init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


class SACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def critic_loss_fn(critic_params):
            next_dist = agent.actor(batch["next_observations"])
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)

            next_q1, next_q2 = agent.target_critic(batch["next_observations"], next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = batch["rewards"] + agent.config["discount"] * batch["masks"] * next_q

            if agent.config["backup_entropy"]:
                target_q = target_q - agent.config["discount"] * batch["masks"] * next_log_probs * agent.temp()

            q1, q2 = agent.critic(batch["observations"], batch["actions"], params=critic_params)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
            }

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch["observations"], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)

            q1, q2 = agent.critic(batch["observations"], actions)
            q = jnp.minimum(q1, q2)

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -1 * log_probs.mean(),
            }

        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                "temp_loss": temp_loss,
                "temperature": temperature,
            }

        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config["target_update_rate"])
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        temp_loss_fn = functools.partial(
            temp_loss_fn, entropy=actor_info["entropy"], target_entropy=agent.config["target_entropy"]
        )
        new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn, has_aux=True)

        return agent.replace(
            rng=new_rng, critic=new_critic, target_critic=new_target_critic, actor=new_actor, temp=new_temp
        ), {**critic_info, **actor_info, **temp_info}

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temp_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = None,
    backup_entropy: bool = True,
    **kwargs,
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims,
        action_dim=action_dim,
        log_std_min=-10.0,
        state_dependent_std=True,
        tanh_squash_distribution=True,
        final_fc_init_scale=1.0,
    )

    actor_params = actor_def.init(actor_key, observations)["params"]
    actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))

    critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
    critic_params = critic_def.init(critic_key, observations, actions)["params"]
    critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
    target_critic = TrainState.create(critic_def, critic_params)

    temp_def = Temperature()
    temp_params = temp_def.init(rng)["params"]
    temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))

    if target_entropy is None:
        target_entropy = -0.5 * action_dim

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=tau,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
        )
    )

    return SACAgent(rng, critic=critic, target_critic=target_critic, actor=actor, temp=temp, config=config)


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict({
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "temp_lr": 3e-4,
        "hidden_dims": (256, 256),
        "discount": 0.99,
        "tau": 0.005,
        "target_entropy": ml_collections.config_dict.placeholder(float),
        "backup_entropy": True,
    })
