from typing import Callable, Any
from jax.random import PRNGKey
from stanza.envs import Environment
from stanza.dataclasses import dataclass, field, replace
from stanza.dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from stanza.goal_conditioned import GCState
from stanza.data import Data
import chex

Action = Any
EnvState = Any 
Noiser = Callable[[PRNGKey, Any, int],Any]
#array of ints
GC_Ind_Sampler = Callable[[PRNGKey, ], tuple]




def last_state_sampler(traj : Data, 
                       target_time : int, 
                       noise_rng_key : PRNGKey,
                       state_noiser : Noiser = None):
    idx = traj.advance(traj.start, target_time)
    start_state = traj.get(idx).observation
    #split key?
    start_state = state_noiser(noise_rng_key,start_state,0) \
        if state_noiser is not None else start_state
    
    return start_state


def roll_in_sampler(traj : Data, 
                    target_time : int,
                    noise_rng_key: PRNGKey, 
                    roll_len : int, 
                    env: Environment,  
                    env_rng_key: PRNGKey,
                    action_noiser: Noiser = None, 
                    process_noiser : Noiser = None ):
    
    """
    @param target_time time of state we wish to sample for goal
    @param roll_len  rollout_length till we hit that time
    """
    if action_noiser is not None or process_noiser is not None:
        assert noise_rng_key is not None
    if action_noiser is None and process_noiser is None:
        return traj.get(target_time).observation
    # we should set the environment state appropriate
    start_time = target_time - roll_len
    start_index = traj.advance(traj.start, start_time)
    start = traj.get(start_index)
    curr_state = start.observation 

    def step(timestep,loop_state):
        env_state, idx, noise_rng, env_rng = loop_state
        action = traj.get(idx).action
        idx = traj.next(idx)
        action_rng, state_rng, noise_rng = jax.random.split(noise_rng, 3) \
            if noise_rng is not None else (None, None, None)
        
        step_rng, env_rng = jax.random.split(env_rng) \
            if env_rng is not None else (None, None)

        action = action_noiser(action_rng, action, timestep) \
            if action_noiser is not None else action
        env_state = env.step(env_state,action, step_rng)
        env_state =  process_noiser(state_rng, env_state, timestep) \
            if process_noiser is not None else env_state
        new_loop_state = (env_state, idx, noise_rng, env_rng)

        # checks consistency of object type
        chex.assert_trees_all_equal_shapes_and_dtypes(loop_state,new_loop_state)
        
    init_state = (curr_state, start, noise_rng_key, env_rng_key, 0)
    end_loop_state = jax.lax.fori_loop(0,roll_len,step,init_state)
    start_state, idx =  end_loop_state[0], end_loop_state[0]
    return start_state, traj.get(idx).action



def roll_in_goal_state_sampler(trajs):
    return #returns a GCState


    

