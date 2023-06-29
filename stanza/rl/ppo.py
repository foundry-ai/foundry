from typing import Any
from stanza.dataclasses import dataclass, field, replace


from jax.random import PRNGKey
from stanza.util.random import PRNGSequence
from stanza.train import Trainer

import jax
import jax.numpy as jnp

import stanza.policies as policies

from stanza.rl import ACPolicy
from stanza.envs import Environment
from stanza.util import extract_shifted
from stanza.data import Data

from stanza import Partial
from typing import Callable
from stanza.util.logging import logger

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
    value: jnp.ndarray
    prev_state: Any
    prev_action: Any
    state: Any

def step_with_reset_(env, state, action, rng):
    d = env.done(state)
    state = jax.lax.cond(d,
        lambda: env.reset(rng),
        lambda: env.step(state, action, rng))
    return state

@dataclass(jax=True)
class PPO:
    gamma: float = 0.9
    num_envs: int = field(default=8, jax_static=True)
    timesteps: int = field(default=10, jax_static=True)
    total_timesteps: int = field(default=5e7, jax_static=True)
    update_epochs: int = field(default=4, jax_static=True)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    trainer: Trainer = field(default_factory=lambda: Trainer())

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
                value=roll.info.value,
                prev_state=earlier_xs,
                prev_action=roll.actions,
                state=later_xs,
            )
            return transitions
        
        rngs = jax.random.split(next(rng), self.num_envs)
        transitions = jax.vmap(rollout)(rngs, state.env_states)
        # extract the final states to use as the new env_states
        env_states = jax.tree_map(lambda x: x[:,-1], transitions.state)
        
        return replace(state,
            rng_key=next_key, 
            env_states=env_states
        ), transitions
    
    def calculate_gae(self, state, transitions):
        last_obs = jax.tree_map(lambda x: x[-1], transitions.state)
        _, last_val = jax.vmap(state.ac_apply, in_axes=(None, 0))(
            state.ac_params, last_obs
        )
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
    
    def loss_fn(self, ac_apply, ac_params, _ac_states, _rng_key, sample):
        transition, gae, targets = sample
        pi, value = ac_apply(ac_params, transition.prev_state)
        log_prob = pi.log_prob(transition.prev_action)
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
        entropy = 0 #pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.vf_coef * value_loss
            - self.ent_coef * entropy
        )
        return None, total_loss, {
            "actor_loss": loss_actor,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss
        }

    def update(self, state, _):
        state, transitions = self.rollout_batch(state)
        advantages, targets = self.calculate_gae(state, transitions)
        transitions, advantages, targets = jax.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            (transitions, advantages, targets)
        )
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
        return state, _

    def train(self, rng_key, env, actor_critic_apply, init_params, *,
              init_opt_state=None):
        if init_opt_state is None:
            init_opt_state = self.trainer.optimizer.init(init_params)

        rngs = jax.random.split(rng_key, self.num_envs)
        env_states = jax.vmap(env.reset)(rngs)
        state = PPOState(
            rng_key=rng_key,
            ac_apply=Partial(actor_critic_apply),
            ac_params=init_params,
            env=env,
            env_states=env_states,
            opt_state=init_opt_state
        )
        update = Partial(type(self).update, self)
        num_updates = self.total_timesteps // self.timesteps // self.num_envs
        state, _ = jax.lax.scan(update, state, (), length=num_updates)
        return state.ac_params