from stanza.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from stanza.policies import PolicyInput, PolicyOutput
from stanza.envs import Environment
from stanza.util import LoopState, extract_shifted
from stanza.util.random import PRNGSequence

from typing import Callable, Any

import jax
import jax.numpy as jnp
import stanza.policies as policies
from jax.random import PRNGKey

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable
    observation_normalizer: Callable = None
    action_normalizer: Callable = None

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        observation = input.observation
        if self.observation_normalizer is not None:
            observation = self.observation_normalizer.normalize(observation)
        pi, value = self.actor_critic(observation)
        action = pi.sample(input.rng_key)
        log_prob = pi.log_prob(action)
        return PolicyOutput(
            action, log_prob, 
            AttrMap(log_prob=log_prob, value=value)
        )
    
@dataclass(jax=True)
class EpisodeState:
    timestep: int
    env_state: Any

@dataclass(jax=True)
class EpisodicEnvironment(Environment):
    env: Environment
    episode_length : jnp.array

    def sample_action(self, rng_key):
        return self.env.sample_action(rng_key)

    def sample_state(self, rng_key):
        return EpisodeState(0, self.env.sample_state(rng_key))

    def reset(self, rng_key):
        return EpisodeState(0, self.env.reset(rng_key))
    
    def step(self, state, action, rng_key):
        def step(state, action, rng_key):
            env_state = self.env.step(
                state.env_state, action, rng_key)
            return EpisodeState(
                timestep=state.timestep + 1,
                env_state=env_state
            )
        def reset(state, action, rng_key):
            env_state = self.env.reset(rng_key)
            return EpisodeState(
                timestep=0,
                env_state=env_state
            )
        done = jnp.logical_or(
            state.timestep == self.episode_length,
            self.env.done(state.env_state)
        )
        return jax.lax.cond(done, reset, 
                step, state, action, rng_key)
    
    def observe(self, state):
        return super().observe(state.env_state)

    def reward(self, state, action, next_state):
        # if we are starting a new episode, 
        # return 0 reward for this step
        new_episode = self.env.done(state.env_state)
        return jax.lax.cond(
            new_episode,
            lambda: jnp.zeros(()),
            lambda: self.env.reward(state.env_state, action, next_state.env_state)
        )

    def done(self, state):
        return jnp.logical_or(self.env.done(state.env_state),
            self.episode_length == state.timestep)
    
    def render(self, state, **kwargs):
        return self.env.render(state.env_state, **kwargs)

@dataclass
class RLState(LoopState):
    rng_key : PRNGKey
    env: Environment
    env_states: Any
    # total rewards of the current episodes
    env_total_rewards: jnp.array
    total_episodes: int
    average_reward: float

@dataclass(jax=True)
class Transition:
    done: jnp.ndarray
    reward: jnp.ndarray
    policy_info: Any
    state: Any
    action: Any
    next_state: Any

@dataclass(jax=True)
class RLAlgorithm:
    num_envs: int = field(default=2048, jax_static=True)
    steps_per_update: int = field(default=10, jax_static=True)

    def compute_stats(self, state):
        stats = {}
        stats["total_timesteps"] = state.iteration * \
                        self.num_envs * self.steps_per_update
        stats["average_reward"] = state.average_reward
        return stats

    def rollout(self, state, policy):
        next_key, rng_key = jax.random.split(state.rng_key)
        rng = PRNGSequence(rng_key)

        def roll(rng_key, x0):
            rng = PRNGSequence(rng_key)
            roll = policies.rollout(state.env.step, x0, policy,
                            model_rng_key=next(rng),
                            policy_rng_key=next(rng),
                            observe=state.env.observe,
                            length=self.steps_per_update)
            xs = roll.states
            earlier_xs, later_xs = extract_shifted(xs)
            us = roll.actions
            reward = jax.vmap(state.env.reward)(earlier_xs, us, later_xs)
            transitions = Transition(
                done=jax.vmap(state.env.done)(later_xs),
                reward=reward,
                policy_info=roll.info,
                state=earlier_xs,
                action=roll.actions,
                next_state=later_xs,
            )
            last_state = jax.tree_map(lambda x: x[-1], xs)
            return transitions, last_state

        rngs = jax.random.split(next(rng), self.num_envs)
        transitions, env_states = jax.vmap(roll)(rngs, state.env_states)

        transitions_reshaped = jax.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        # compute new reward totals
        def total_reward_scan(total_reward, transition):
            total_reward = total_reward + transition.reward
            carry = transition.done * total_reward
            return carry, total_reward
        env_total_rewards, total_rewards = jax.lax.scan(total_reward_scan, 
                                        state.env_total_rewards, transitions_reshaped)
        finished_episodes = jnp.count_nonzero(transitions_reshaped.done)
        new_rewards = jnp.sum(total_rewards*transitions_reshaped.done)

        avg = new_rewards / jnp.maximum(1, finished_episodes)

        frac = state.total_episodes / (jnp.maximum(state.total_episodes, 1) + finished_episodes)

        average_reward = frac*state.average_reward + avg * (1 - frac)
        # update the average reward
        state = replace(state, 
            rng_key=next_key, 
            env_total_rewards=env_total_rewards,
            average_reward=average_reward,
            total_episodes=state.total_episodes + finished_episodes,
            env_states=env_states)
        return state, transitions

    def init(self, rng_key, env, max_iterations, rl_hooks):
        rngs = jax.random.split(rng_key, self.num_envs)
        env_states = jax.vmap(env.reset)(rngs)
        state = RLState(
            iteration=0,
            max_iterations=max_iterations,
            hooks=rl_hooks,
            hook_states=None,
            rng_key=rng_key,
            env=env,
            env_states=env_states,
            env_total_rewards=jnp.zeros(self.num_envs),
            total_episodes=0,
            average_reward=0,
            last_stats=None
        )
        return state