from stanza.dataclasses import dataclass, field, replace, combine
from stanza.util.attrdict import AttrMap
from stanza.policies import PolicyInput, PolicyOutput
from stanza.envs import Environment
from stanza.util.loop import LoopState, init_hooks as _init_hooks
from stanza.util import extract_shifted
from stanza.util.random import PRNGSequence

from typing import Callable, List, Any

import jax
import jax.numpy as jnp
import stanza.policies as policies
from jax.random import PRNGKey

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable
    obs_normalizer: Callable = None
    action_normalizer: Callable = None
    use_mean : bool = field(default=False, jax_static=True)
    

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        observation = input.observation
        if self.obs_normalizer is not None:
            observation = self.obs_normalizer.normalize(observation)
        pi, value = self.actor_critic(observation)
        if self.use_mean:
            action = pi.mean
        else:
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
    episode_length : jnp.array = field(jax_static=True)

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
        return self.env.observe(state.env_state)

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


@dataclass(jax=True)
class RLConfig:
    rng_key: PRNGKey = None
    env: Environment = None
    num_envs: int = field(default=2048, jax_static=True)
    total_timesteps: int = 1_000_000
    steps_per_iteration: int = field(default=10, jax_static=True)

    num_eval: int = field(default=8, jax_static=True)
    episode_length: int = field(default=None, jax_static=True)

    rl_hooks: List[Callable] = field(default_factory=list)

@dataclass(jax=True)
class RLState(LoopState):
    config : RLConfig
    rng_key : PRNGKey
    # the episodic states
    episode_states: Any
    # total rewards of the current episodes
    env_total_rewards: jnp.array
    total_episodes: int

@dataclass(jax=True)
class Transition:
    done: jnp.ndarray
    reward: jnp.ndarray
    policy_info: Any
    obs: Any
    action: Any
    next_obs: Any

class RLAlgorithm:
    @staticmethod
    def evaluate(state, policy, rng_key):
        env = state.config.env
        def roll(rng_key):
            x0_rng, prk, mrk = jax.random.split(rng_key, 3)
            x0 = env.reset(x0_rng)
            traj = policies.rollout(env.step, x0, policy,
                policy_rng_key=prk, model_rng_key=mrk,
                length=state.config.episode_length,
                observe=state.config.env.observe)
            earlier_xs, later_xs = extract_shifted(traj.states)
            us = traj.actions
            rewards = jax.vmap(env.reward)(earlier_xs, us, later_xs)
            return jnp.sum(rewards)
        rngs = jax.random.split(rng_key, state.config.num_eval)
        return jnp.mean(jax.vmap(roll)(rngs))

    @staticmethod
    def rollout(state, policy):
        next_key, rng_key = jax.random.split(state.rng_key)
        rng = PRNGSequence(rng_key)

        episode_env = EpisodicEnvironment(
            state.config.env,
            state.config.episode_length
        )

        def roll(rng_key, x0):
            rng = PRNGSequence(rng_key)
            roll = policies.rollout(episode_env.step, x0, policy,
                            model_rng_key=next(rng),
                            policy_rng_key=next(rng),
                            observe=episode_env.observe,
                            length=state.config.steps_per_iteration)
            obs = roll.observations
            earlier_obs, later_obs = extract_shifted(obs)
            us = roll.actions
            reward = jax.vmap(episode_env.reward)(earlier_obs, us, later_obs)
            transitions = Transition(
                done=jax.vmap(episode_env.done)(later_obs),
                reward=reward,
                policy_info=roll.info,
                obs=earlier_obs,
                action=roll.actions,
                next_obs=later_obs,
            )
            last_state = jax.tree_map(lambda x: x[-1], roll.states)
            return transitions, last_state


        rngs = jax.random.split(next(rng), state.config.num_envs)
        transitions, episode_states = jax.vmap(roll)(rngs, state.episode_states)

        transitions_reshaped = jax.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        finished_episodes = jnp.count_nonzero(transitions_reshaped.done)
        state = replace(state, 
            rng_key=next_key, 
            total_episodes=state.total_episodes + finished_episodes,
            episode_states=episode_states)
        return state, transitions
    
    @staticmethod
    def init(config, init_hooks=True):
        rng_key, tk = jax.random.split(config.rng_key)
        episodic_env = EpisodicEnvironment(
            config.env, config.episode_length
        )
        episode_states = jax.vmap(episodic_env.reset)(
            jax.random.split(tk, config.num_envs)
        )
        iterations = (config.total_timesteps // config.steps_per_iteration) // config.num_envs

        rl_state = RLState(
            iteration=0,
            max_iterations=iterations,
            hooks=config.rl_hooks,
            hook_states=None,
            last_stats=None,

            config=config,
            rng_key=rng_key,
            episode_states=episode_states,
            env_total_rewards=jnp.zeros(config.num_envs),
            total_episodes=0,
        )
        if init_hooks:
            rl_state = _init_hooks(rl_state)
        return rl_state