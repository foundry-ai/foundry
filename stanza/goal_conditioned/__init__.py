import jax
import jax.numpy as jnp
from stanza.dataclasses import dataclass, field, replace
from stanza.envs import Environment
from typing import Callable, Any
from jax.random import PRNGKey

Goal = Any
EnvState = Any
Action = Any
# a (goal,state tuple).
# wraps a goal
@dataclass(jax = True)
class GCState:
    goal: Goal
    env_state: EnvState


@dataclass(jax = True)
class EndGoal:
    end_state: EnvState
    other_info: Any = None

@dataclass(jax = True)
class GCObs:
    goal: Goal
    env_obs: Any

GoalAndStateSampler = Callable[[PRNGKey], GCState]
GoalReward = Callable[[GCState,Action,EnvState],float]
GoalDoneFunc = Callable[[GCState],bool]

@dataclass(jax = True)
class GCEnvironment(Environment):
    env: Environment
    gs_sampler: GoalAndStateSampler
    gc_reward: GoalReward
    g_done: GoalDoneFunc
   
    def sample_action(self, rng_key):
        return self.env.sample_action(rng_key)
    
    def sample_state(self, rng_key):
        return self.reset(rng_key)
    
    def reset(self, rng_key):
        return self.gs_sampler(rng_key)
    
    def step(self, state, action, rng_key):
        new_env_state = self.env.step(state.env_state,action,rng_key)
        new_gc_state = GCState(state.goal,new_env_state)
        return new_gc_state
    
    def observe(self, state):
        env_obvs = self.env.observe(state.env_state)
        return GCObs(state.goal, env_obvs)
    
    def reward(self,state,action,next_state):
        return self.gc_reward(state,action,next_state.env_state)

    def done(self,state):
        return jnp.logical_or(self.env.done(state.env_state),
                              self.g_done(state))

    def render(self,state,**kwargs):
        return self.env.render(state.env_state, **kwargs)


# notice that most methods go unimplemented
@dataclass(jax=True)
class GCHighLevelEnvironment(Environment):
    gs_sampler: GoalAndStateSampler   
    base_env : Environment  = None #for rendering   
    
    def sample_action(self, rng_key : PRNGKey):
        return self.gs_sampler(rng_key).goal

    def sample_state(self, rng_key : PRNGKey):
        # note, we use the state_high_level method because its observation is the state
        return self.gs_sampler(rng_key).env_state
    
    # the step in the high level just goes to the 
    #target
    def step(self, state, action, rng_key):
        # requires thats action is a goal
        return action.end_state
    
    def render(self, state, width=256, height=256):
        return self.base_env.render(state, width, height)
    
    #notice state space is equal to state space of original env

    #default observe: returns the state