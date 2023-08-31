from typing import Any
from stanza.dataclasses import dataclass, field, replace, unpack, combine

from jax.random import PRNGKey
from stanza.train import Trainer, TrainState, TrainResults, TrainConfig
from stanza.util.attrdict import AttrMap
import stanza.util.loop as loop

import jax
import jax.numpy as jnp

import stanza.policies as policies
import stanza.util

from stanza.data import Data

from stanza import Partial
from typing import Callable
from stanza.util.logging import logger
from stanza.rl import (
    RLState, Transition, RLAlgorithm, 
    RLConfig, ACPolicy, EpisodicEnvironment
)
from stanza.data.normalizer import Normalizer

import optax

@dataclass(jax=True)
class PPOState(RLState):
    obs_normalizer: Normalizer
    reward_normalizer: Normalizer
    train_state : TrainState

@dataclass(jax=True)
class PPOConfig(RLConfig):
    ac_apply: Callable = None
    obs_normalizer: Normalizer = None
    reward_normalizer: Normalizer = None

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5

    epochs_per_iteration: int = 4

    trainer: Trainer = None

    # passed as args to the trainer
    init_params: Any = None
    init_state: Any = None
    init_opt_state: Any = None

    optimizer: optax.GradientTransformation = optax.adam(1e-3)
    batch_size: int = field(default=32, jax_static=True)

@dataclass(jax=True)
class PPO(RLAlgorithm, PPOConfig):
    def compute_stats(self, state):
        rng_key, sk = jax.random.split(state.rng_key)
        state = replace(state, rng_key=rng_key)
        ac_apply = Partial(state.ac_apply, state.train_state.fn_params)
        policy = ACPolicy(ac_apply)
        stats = {
            "timesteps": state.iteration * state.config.num_envs \
                            * state.config.steps_per_iteration,
            "episode_reward": self.evaluate(state, policy, sk)
        }
        return state, stats
    
    def calculate_gae(self, state, transitions):
        last_obs = jax.tree_map(lambda x: x[-1], transitions.obs)
        if state.obs_normalizer is not None:
            last_obs = state.obs_normalizer.normalize(last_obs)
        _, last_val = jax.vmap(state.ac_apply, in_axes=(None, 0))(
            state.train_state.fn_params, last_obs
        )
        def _calc_advantage(gae_and_nv, transition):
            gae, next_val = gae_and_nv
            done, value, reward = (
                transition.done,
                transition.policy_info.value,
                transition.reward
            )
            if state.reward_normalizer is not None:
                reward = state.reward_normalizer.normalize(reward)
            delta = reward + self.gamma * (1 - done) * next_val - value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_calc_advantage,
            (jnp.zeros_like(last_val), last_val),
            transitions,
            reverse=True
        )
        return advantages, advantages + transitions.policy_info.value
    
    @staticmethod
    def loss_fn(config, ac_apply, normalizers, ac_params, _rng_key, batch):
        obs_norm, _ = normalizers
        transition, gae, targets = batch
        obs = transition.obs
        if obs_norm is not None:
            obs = obs_norm.normalize(obs)
        pi, value = jax.vmap(ac_apply, in_axes=(None, 0))(ac_params, obs)
        # vmap the log_prob function over the pi, prev_action batch
        log_prob = jax.vmap(type(pi).log_prob)(pi, transition.action)
        value_pred_clipped = transition.policy_info.value + (
            value - transition.policy_info.value
        ).clip(-config.clip_eps, config.clip_eps)
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
                1.0 - config.clip_eps,
                1.0 + config.clip_eps
            ) * gae )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = jax.vmap(type(pi).entropy)(pi).mean()

        total_loss = (
            loss_actor
            + config.vf_coef * value_loss
            - config.ent_coef * entropy
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
        # update the normalizers
        obs_normalizer = state.obs_normalizer.update(transitions.next_obs) \
            if state.obs_normalizer is not None else None
        reward_normalizer = state.reward_normalizer.update(transitions.reward) \
            if state.reward_normalizer is not None else None
        state = replace(state,
            obs_normalizer=obs_normalizer,
            reward_normalizer=reward_normalizer
        )
        data = Data.from_pytree((transitions, advantages, targets))
        # reset the train state to make it continue training
        train_state = state.train_state
        train_state = replace(train_state,
             # update the normalizers for training
            fn_state=(state.obs_normalizer, 
                      state.reward_normalizer, train_state.fn_state[2]),
            iteration=0,
            max_iterations=state.config.epochs_per_iteration * data.length // state.config.batch_size,
            epoch_iteration=0, epoch=0)
        train_state = state.config.trainer.run(train_state, data)

        state = replace(
            state,
            iteration=state.iteration + 1,
            train_state=train_state,
        )
        state, stats = self.compute_stats(state)
        state = replace(state,
            last_stats=stats
        )
        state = loop.run_hooks(state)
        return state

    def init(self, config=None, init_hooks=True, **kwargs):
        if config is None:
            config = self
        config = combine(PPOConfig, self, kwargs)
        rl_state = super().init(config, init_hooks=False)

        actor_critic_apply = Partial(config.ac_apply)
        rng_key, tk = jax.random.split(rl_state.rng_key)

        loss_fn = Partial(self.loss_fn, config, actor_critic_apply)

        # sample a datapoint to initialize the trainer
        episodic_env = EpisodicEnvironment(config.env, config.episode_length)
        sample_action = episodic_env.sample_action(PRNGKey(42))
        sample_obs = episodic_env.observe(episodic_env.sample_state(PRNGKey(42)))
        sample = Transition(
            done=jnp.array(True),
            reward=jnp.zeros(()),
            policy_info=AttrMap(log_prob=jnp.zeros(()),value=jnp.zeros(())),
            obs=sample_obs, action=sample_action, next_obs=sample_obs
        ), jnp.zeros(()), jnp.zeros(())
        train_state = self.trainer.init(sample,
            loss_fn=loss_fn,rng_key=tk,max_iterations=0,
            init_params=config.init_params,
            init_state=(config.obs_normalizer, config.reward_normalizer,
                        config.init_state),
            init_opt_state=config.init_opt_state
        )
        state = PPOState(
            **unpack(rl_state),
            train_state=train_state,
            obs_normalizer=config.obs_normalizer,
            reward_normalizer=config.reward_normalizer
        )
        state, stats = self.compute_stats(state)
        state = replace(state, last_stats=stats, rng_key=rng_key)
        if init_hooks:
            state = stanza.util.loop.init_hooks(state)
        return state
    
    def run(self, state):
        update = Partial(type(self).update, self)
        state = loop.run_hooks(state)
        state = loop.loop(update, state)
        return state

    def train(self, **kwargs):
        config = combine(PPOConfig, self, kwargs)
        print(jax.tree_map(lambda x: x.shape, config))
        with jax.profiler.TraceAnnotation("rl"):
            state = self.init(config)
            state = self.run(state)
        return TrainResults(
            fn_params=state.train_state.fn_params,
            fn_state=state.train_state.fn_state,
            opt_state=state.train_state.opt_state,
            hook_states=None
        )