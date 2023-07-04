from typing import Any
from stanza.dataclasses import dataclass, field, replace

from jax.random import PRNGKey
from stanza.util.random import PRNGSequence
from stanza.train import Trainer, TrainState, TrainResults

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
from stanza.util import LoopState, loop

@dataclass(jax=True)
class PPOState(LoopState):
    rng_key : PRNGKey
    ac_apply : Callable
    train_state : TrainState
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
    gamma: float = 0.99
    total_timesteps: int = 1e7
    num_envs: int = field(default=2048, jax_static=True)
    timesteps: int = field(default=10, jax_static=True)
    update_epochs: int = field(default=4, jax_static=True)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    trainer: Trainer = field(
        default_factory=lambda: Trainer(batch_size=512)
    )

    def calculate_stats(self, state):
        stats = dict(state.train_state.last_stats)
        stats["total_timesteps"] = state.iteration * self.num_envs * self.timesteps
        return stats

    def rollout_batch(self, state):
        next_key, rng_key = jax.random.split(state.rng_key)
        rng = PRNGSequence(rng_key)
        ac = Partial(state.ac_apply, state.train_state.fn_params)
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
            state.train_state.fn_params, last_obs
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
    
    def loss_fn(self, ac_apply, _ac_states, ac_params, _rng_key, sample):
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

    def update(self, state):
        with jax.profiler.StepTraceAnnotation("rollout_batch", step_num=state.iteration):
            state, transitions = self.rollout_batch(state)
        with jax.profiler.StepTraceAnnotation("calculate_gae", step_num=state.iteration):
            advantages, targets = self.calculate_gae(state, transitions)
        transitions, advantages, targets = jax.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            (transitions, advantages, targets)
        )
        data = Data.from_pytree((transitions, advantages, targets))
        # reset the train state
        # to make it continue training
        train_state = replace(state.train_state,
            iteration=0,
            max_iterations=None,
            epoch_iteration=0,
            max_epochs=self.update_epochs,
            epoch=0)
        with jax.profiler.StepTraceAnnotation("train", step_num=state.iteration):
            train_state = self.trainer.run(train_state, data)
        state = replace(
            state,
            iteration=state.iteration + 1,
            #train_state=train_state,
            last_stats=self.calculate_stats(state)
        )
        with jax.profiler.StepTraceAnnotation("hooks", step_num=state.iteration):
            state = stanza.util.run_hooks(state)
        return state

    
    def init(self, rng_key, env, actor_critic_apply, init_params,
             *, init_opt_state=None, rl_hooks=[], train_hooks=[]):
        actor_critic_apply = Partial(actor_critic_apply)
        rng_key, tk = jax.random.split(rng_key)
        rngs = jax.random.split(rng_key, self.num_envs)
        env_states = jax.vmap(env.reset)(rngs)
        num_updates = (self.total_timesteps // self.timesteps) // self.num_envs

        loss_fn = Partial(type(self).loss_fn, self, actor_critic_apply)
        # sample a datapoint to initialize the trainer
        sample = Transition(
            done=jnp.array(True),
            reward=jnp.zeros(()),
            log_prob=jnp.zeros(()),
            value=jnp.zeros(()),
            prev_state=env.sample_state(PRNGKey(42)),
            prev_action=env.sample_action(PRNGKey(42)),
            state=env.sample_state(PRNGKey(42))
        ), jnp.zeros(()), jnp.zeros(())

        train_state = self.trainer.init(loss_fn, sample, 0,
                                        tk, init_params, init_opt_state=init_opt_state,
                                        hooks=train_hooks)
        state = PPOState(
            iteration=0,
            max_iterations=num_updates,
            hooks=rl_hooks,
            hook_states=None,
            last_stats=None,
            rng_key=rng_key,
            env=env,
            env_states=env_states,
            ac_apply=actor_critic_apply,
            train_state=train_state
        )
        state = replace(state, last_stats=self.calculate_stats(state))
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
        with jax.profiler.TraceAnnotation("ppo"):
            state = self.init(rng_key, env, actor_critic_apply, init_params,
                            init_opt_state=init_opt_state, 
                            rl_hooks=rl_hooks, train_hooks=train_hooks)
            state = self.run(state)
        return TrainResults(
            fn_params=state.train_state.fn_params,
            fn_state=state.train_state.fn_params,
            opt_state=state.train_state.opt_state,
            hook_states=None
        )