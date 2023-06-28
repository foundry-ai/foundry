from typing import Any
from stanza.dataclasses import dataclass, field, replace

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from jax.random import PRNGKey
from stanza.util.random import PRNGSequence
from stanza.train import Trainer

import jax
import jax.numpy as jnp
import distrax
import stanza.policies as policies
from stanza.envs import Environment

from stanza.policies import PolicyInput, PolicyOutput
from stanza.util.attrdict import AttrMap
from stanza.util import extract_shifted
from stanza.data import Data

from stanza import Partial, partial
from typing import Sequence, Callable

class DenseActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        pi, value = self.actor_critic(input.observation)
        action = pi.sample(input.rng_key)
        log_prob = pi.log_prob(action)
        return PolicyOutput(
            action, log_prob, 
            AttrMap(log_prob=log_prob, value=value)
        )
    

@dataclass(jax=True)
class PPOState:
    rng_key : PRNGKey
    ac_apply : Callable
    ac_params : Any
    # optimizer state
    opt_state : Any

    env: Environment
    env_states: Any

@dataclass(jax=True)
class Transition:
    done: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    prev_state: Any
    prev_action: Any
    state: Any

def step_with_reset_(env, state, action, rng):
    d = env.done(state)
    state = jax.lax.cond(d,
        lambda _: env.reset(rng),
        lambda _: env.step(state, action, rng))
    return state

@dataclass(jax=True)
class PPO:
    gamma: float = 0.9
    timesteps: int = field(default=10, jax_static=True)
    total_timesteps: int = field(default=5e7, jax_static=True)
    update_epochs: int = field(default=4, jax_static=True)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    trainer: Trainer = None

    def rollout_batch(self, state):
        next_key, rng_key = jax.random.split(state.rng_key)
        rng = PRNGSequence(rng_key)
        ac = Partial(state.ac_apply, state.ac_params)
        ac_policy = ACPolicy(ac)

        def rollout(rng_key, x0):
            rng = PRNGSequence(rng_key)
            # resets the environment when done
            step = Partial(step_with_reset_, state.env)
            roll = policies.rollout(step, x0, ac_policy,
                            model_rng_key=next(rng),
                            policy_rng_key=next(rng),
                            length=self.timesteps)
            xs = roll.states
            earlier_xs, later_xs = extract_shifted(xs)
            us = roll.actions
            reward = jax.vmap(state.env.reward)(earlier_xs, us, later_xs)
            transitions = Transition(
                done=jax.vmap(state.env.done)(later_xs),
                reward=reward,
                log_prob=roll.info.log_prob,
                prev_state=earlier_xs,
                prev_action=roll.actions,
                state=later_xs,
            )
            return transitions
        
        transitions = jax.vmap(rollout)(rng, state.env_states)
        # extract the final states to use as the new env_states
        env_states = jax.tree_map(lambda x: x[:,-1], transitions.state)
        
        return replace(state,
            rng_key=next_key, 
            env_states=env_states
        ), transitions
    
    def calculate_gae(self, state, transitions):
        last_obs = jax.tree_map(lambda x: x[-1], transitions.state)
        _, last_val = state.ac_apply(state.ac_params, last_obs)
        def _calc_advantage(gae_and_nv, transition):
            gae, next_val = gae_and_nv
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward
            )
            delta = reward + self.gamma * (1 - done) * next_val - value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_calc_advantage, 
            (jnp.zeros_like(last_val), last_val),
            transitions,
            reverse=True
        )
        return advantages, advantages + transitions.value
    
    def loss_fn(self, ac_apply, ac_params, sample):
        transition, gae, targets = sample
        pi, value = ac_apply(ac_params, transition.prev_state)
        log_prob = pi.log_prob(transition.action)
        value_pred_clipped = transition.value + (
            value - transition.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        ratio = jnp.exp(log_prob - transition.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps
            ) * gae )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.vs_coef * value_loss
            - self.ent_coef * entropy
        )
        return total_loss, {
            "actor_loss": loss_actor,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss
        }

    def update(self, state):
        state, transitions = self.rollout_batch(state)
        advantages, targets = self.calculate_gae(state, transitions)
        data = Data.from_pytree((transitions, advantages, targets))
        loss_fn = Partial(type(self).loss_fn, 
            self, state.ac_apply)
        rng_key, sk = jax.random.split(state.rng_key)
        result = self.trainer.train(loss_fn, data, sk,
                            state.ac_params, None,
                            init_opt_state=state.opt_state,
                            epochs=self.update_epochs)
        opt_state, ac_params = result.opt_state, result.fn_params

        state = replace(
            state,
            rng_key=rng_key,
            ac_params=ac_params,
            opt_state=opt_state
        )
        return state

    def train(self, env, actor_critic_apply, init_params, *,
              init_opt_state=None):
        if init_opt_state is None:
            init_opt_state = self.trainer.optimizer.init(init_params)
        state = PPOState(
            ac_apply=actor_critic_apply,
            env=env,
            opt_state=init_opt_state
        )
        update = Partial(type(self).update, self)
        num_updates = self.total_timesteps // self.num_steps // self.num_envs
        state, _ = jax.lax.scan(update, state, (), length=num_updates)
        return state.ac_params