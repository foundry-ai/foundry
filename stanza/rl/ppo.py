from typing import Any
from stanza.dataclasses import dataclass, field, replace, unpack

from jax.random import PRNGKey
from stanza.util.random import PRNGSequence
from stanza.train import Trainer, TrainState, TrainResults
from stanza.util.attrdict import AttrMap

import jax
import jax.numpy as jnp

import stanza.policies as policies
import stanza.util

from stanza.rl import ACPolicy
from stanza.envs import Environment
from stanza.util import extract_shifted
from stanza.data import Data

from stanza import Partial
from typing import Callable
from stanza.util.logging import logger
from stanza.rl import RLState, Transition, RLAlgorithm

@dataclass(jax=True)
class PPOState(RLState):
    ac_apply : Callable
    train_state : TrainState

@dataclass(jax=True)
class PPO(RLAlgorithm):
    gamma: float = 0.99
    total_timesteps: int = 1_000_000
    update_epochs: int = field(default=4, jax_static=True)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    trainer: Trainer = field(
        default_factory=lambda: Trainer(batch_size=512)
    )

    def compute_stats(self, state):
        stats = super().compute_stats(state)
        stats.update(state.train_state.last_stats)
        return stats
    
    def calculate_gae(self, state, transitions):
        last_obs = jax.tree_map(lambda x: x[-1], transitions.state)
        _, last_val = jax.vmap(state.ac_apply, in_axes=(None, 0))(
            state.train_state.fn_params, jax.vmap(state.env.observe)(last_obs)
        )
        def _calc_advantage(gae_and_nv, transition):
            gae, next_val = gae_and_nv
            done, value, reward = (
                transition.done,
                transition.policy_info.value,
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
        return advantages, advantages + transitions.policy_info.value
    
    def loss_fn(self, env, ac_apply, _ac_states, ac_params, _rng_key, batch):
        transition, gae, targets = batch

        obs = jax.vmap(env.observe)(transition.state)
        pi, value = jax.vmap(ac_apply, in_axes=(None, 0))(ac_params, obs)
        # vmap the log_prob function over the pi, prev_action batch
        log_prob = jax.vmap(type(pi).log_prob)(pi, transition.action)
        value_pred_clipped = transition.policy_info.value + (
            value - transition.policy_info.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        ratio = jnp.exp(log_prob - transition.policy_info.log_prob)
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
        entropy = jax.vmap(type(pi).entropy)(pi).mean()

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

    def update(self, state):
        # rollout the current policy 
        ac_apply = Partial(state.ac_apply, state.train_state.fn_params)
        policy = ACPolicy(ac_apply)

        state, transitions = self.rollout(state, policy)
        # reshape the transitions so that time is the first axes
        transitions = jax.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        advantages, targets = self.calculate_gae(state, transitions)

        # flatten the transitions, advantages, targets
        transitions, advantages, targets = jax.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            (transitions, advantages, targets)
        )
        data = Data.from_pytree((transitions, advantages, targets))
        # reset the train state
        # to make it continue training
        train_state = state.train_state
        train_state = replace(train_state,
            iteration=0,
            max_iterations=self.update_epochs * data.length // self.trainer.batch_size,
            epoch_iteration=0, epoch=0)
        train_state = self.trainer.run(train_state, data)

        state = replace(
            state,
            iteration=state.iteration + 1,
            train_state=train_state,
        )
        state = replace(state,
            last_stats=self.compute_stats(state)
        )
        state = stanza.util.run_hooks(state)
        return state

    def init(self, rng_key, env,
             actor_critic_apply, init_params,
             *, init_opt_state=None, rl_hooks=[], train_hooks=[]):
        actor_critic_apply = Partial(actor_critic_apply)
        rng_key, tk = jax.random.split(rng_key)
        num_updates = (self.total_timesteps // self.steps_per_update) // self.num_envs
        rl_state = super().init(rng_key, env, num_updates, rl_hooks)
        loss_fn = Partial(type(self).loss_fn, self, env, actor_critic_apply)
        # sample a datapoint to initialize the trainer
        sample = Transition(
            done=jnp.array(True),
            reward=jnp.zeros(()),
            policy_info=AttrMap(log_prob=jnp.zeros(()),value=jnp.zeros(())),
            state=env.sample_state(PRNGKey(42)),
            action=env.sample_action(PRNGKey(42)),
            next_state=env.sample_state(PRNGKey(42))
        ), jnp.zeros(()), jnp.zeros(())

        train_state = self.trainer.init(loss_fn, sample, 0,
                                        tk, init_params, init_opt_state=init_opt_state,
                                        hooks=train_hooks, epochs=self.update_epochs)
        state = PPOState(
            **unpack(rl_state),
            ac_apply=actor_critic_apply,
            train_state=train_state
        )
        state = replace(state, last_stats=self.compute_stats(state))
        state = stanza.util.init_hooks(state)
        return state
    
    def run(self, state):
        update = Partial(type(self).update, self)
        state = stanza.util.run_hooks(state)
        state = stanza.util.loop(update, state)
        return state

    def train(self, rng_key, env, actor_critic_apply, init_params, *,
              init_opt_state=None,
              rl_hooks=[], train_hooks=[]):
        with jax.profiler.TraceAnnotation("rl"):
            state = self.init(rng_key, env, actor_critic_apply, init_params,
                            init_opt_state=init_opt_state, 
                            rl_hooks=rl_hooks, train_hooks=train_hooks)
            state = self.run(state)
        return TrainResults(
            fn_params=state.train_state.fn_params,
            fn_state=state.train_state.fn_state,
            opt_state=state.train_state.opt_state,
            hook_states=None
        )