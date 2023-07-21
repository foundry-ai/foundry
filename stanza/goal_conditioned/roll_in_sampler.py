from typing import Callable, Any
from jax.random import PRNGKey
from stanza.envs import Environment
from stanza.dataclasses import dataclass, field, replace
from stanza.dataclasses import dataclass, field, replace
import jax
import jax.numpy as jnp
from stanza.goal_conditioned import GCState, GCObs
from stanza.data import Data
import chex
from stanza.util.random import PRNGSequence
from stanza.goal_conditioned import EndGoal
from stanza.data.trajectory import Timestep


State = Any
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


from stanza.util import shape_tree
def roll_in_sample(traj : Data, 
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
        obs = traj.get(target_time).observation
        act = traj.get(target_time).action
        return obs, act
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
        env_state = env.step(env_state, action, step_rng)
        env_state =  process_noiser(state_rng, env_state, timestep) \
            if process_noiser is not None else env_state
        new_loop_state = (env_state, idx, noise_rng, env_rng)
        # checks consistency of object type
        chex.assert_trees_all_equal_shapes_and_dtypes(loop_state,new_loop_state)
        return new_loop_state

        
    init_state = (curr_state, start, noise_rng_key, env_rng_key, 0)
    end_loop_state = jax.lax.fori_loop(0,roll_len,step,init_state)
    start_state, idx =  end_loop_state[0], end_loop_state[1]
    return start_state, traj.get(idx).action


#TODO throw away
@dataclass(jax=True)
class GCSampler:
    def sample_gc_state(self, key : PRNGKey):
        raise NotImplementedError("Must impelement sample_gc_state()")
    def sample_gc_timestep(self, key : PRNGKey):
        raise NotImplementedError("Must impelement sample_gc_timestep()")
    def sample_gc_timestep_high_level(self, key : PRNGKey):
        raise NotImplementedError("Must impelement sample_gc_state()")


    
@dataclass(jax=True)
class RollInSampler:
    action_noiser : Noiser = None
    process_noiser : Noiser = None
    traj_data : Any = None
    delta_t_min : int = 3
    delta_t_max : int = 8
    roll_len_min : int = 3
    roll_len_max : int = 8
    env : Environment = None
    fixed_goal : State = None


    def sample_goal_state_action(self, key : PRNGKey):
        rng = PRNGSequence(key)
        rand_traj = self.traj_data.sample(next(rng))
        traj_len = rand_traj.length

        delta_t = jax.random.randint(next(rng), (), minval = self.delta_t_min,
                                     maxval = self.delta_t_max+1)
        delta_t = jax.lax.cond(delta_t <= traj_len - 1, lambda x: x, lambda x: traj_len - 1, operand = delta_t) 
        #print('delta_t', delta_t, 'min', self.delta_t_min, 'max', self.delta_t_max)
        
        start_t = jax.random.randint(next(rng), (), minval = 1,
                                 maxval = traj_len - delta_t)
        
        #print('start_t', start_t)
        roll_len = jax.random.randint(next(rng), (), minval = self.roll_len_min,
                                      maxval = self.roll_len_max)
    
        roll_len = jax.lax.cond(roll_len < start_t + 1, 
                                lambda x: x, lambda x: start_t, operand = roll_len)
        
        start_state, start_action =  roll_in_sample(traj = rand_traj,
                    target_time = start_t,
                    noise_rng_key = next(rng), 
                    roll_len = roll_len, 
                    env = self.env, 
                    env_rng_key = next(rng),
                    action_noiser=self.action_noiser, 
                    process_noiser=self.process_noiser)
        
        
        if self.fixed_goal is None:
            end_state = rand_traj.get(start_t + delta_t).observation
        else:
            end_state = self.fixed_goal
        goal = EndGoal(end_state)
        return start_state, goal, start_action
    
    #@partial(jax.jit, static_argnames=['encode_start'])
    def sample_gc_state(self, key : PRNGKey):
        start_state, goal, _ = self.sample_goal_state_action(key)
        return GCState(goal=goal, 
                            env_state=start_state)

    def sample_gc_timestep(self, key : PRNGKey):
        start_state, goal, start_action = self.sample_goal_state_action(key)
        start_obs = self.env.observe(start_state)
        #goal = end_state 
        #EndGoal(start_state=None, end_state=end_state)
        #goal = StartEndGoal(start_state = None, end_state = end_state)
        return Timestep(observation=(GCObs(goal=goal, 
                                           env_obs=start_obs)),
                                           action=start_action)

    def sample_gc_state_high_level(self, key : PRNGKey):
        start_state, goal, _ = self.sample_goal_state_action(key)
        return Timestep(observation=start_state, action=goal)

    def sample_gc_timestep_high_level(self, key : PRNGKey):
        start_state, goal, _ = self.sample_goal_state_action(key)
        start_obs = self.env.observe(start_state)
        return Timestep(observation=start_obs, action=goal)

    
