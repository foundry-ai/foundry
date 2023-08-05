"""
extracts training data from rollouts
"""

#TODO negative examples 

import jax
from typing import Callable, Any
import chex
from stanza.data import Data
from stanza.dataclasses import dataclass

@dataclass(jax = True)
class BCGCTuple:
    init_timestep : Any
    goal_timestep : Any
    cost_to_go : float
    delta_t : int 

# default value is # of time steps
def time_step_cost_to_go_update(cost_to_go : float, ind: int, timestep):
    return cost_to_go + 1

def extract_data_from_traj_at_idx(traj : Data, start_idx : int,
                           max_time_steps : int - 1,
                           cost_to_go_update : Callable[[float,int,Any],float] 
                           = time_step_cost_to_go_update ):
    

    def inner_continue(ind,loop_state):
        #TODO
        idx, ind,_, _, _ = loop_state
        pol_state = traj.get(idx).pol_state
        return pol_state is not None and \
            pol_state.noise_phase is False \
            and (max_time_steps <= 0 or ind < max_time_steps)
    
    def inner_step(loop_state):
        idx, ind, prev_val, init_timestep, data = loop_state
        curr_timestep = traj.get(idx)
        val = cost_to_go_update(prev_val, ind, curr_timestep)
        new_data = data.append(BCGCTuple(init_timestep=init_timestep, 
                                      goal_timestep=curr_timestep, 
                                      val=val, delta_t=ind))
        
        new_loop_state = (traj.next(idx), ind+1, val, init_timestep, new_data)
        chex.assert_trees_all_equal_shapes_and_dtypes(loop_state,new_loop_state)
        return new_loop_state
    
    init_timestep = traj.get(start_idx)
    init_val = 0
    data = Data()

    init_loop_state = (init_val, start_idx, init_timestep, data)
    end_loop_state = jax.lax.while_loop(cond_fun=inner_continue,
                                        body_fun=inner_step,
                                        init_val=init_loop_state)
    return end_loop_state[4]


def extra_data_from_traj(traj : Data, max_time_steps : int = -1,
                            cost_to_go_update : Callable[[float,int,Any],float]
                            = time_step_cost_to_go_update):
    start_index = traj.start
    all_data = Data()
    init_loop_state = (start_index, all_data)
    def step(timestep, loop_state):
        idx, all_data = loop_state
        step_data = extract_data_from_traj_at_idx(traj=traj,
            max_time_steps=max_time_steps,
                start_idx=idx,
                         cost_to_go_update=cost_to_go_update)
        new_idx = traj.next(idx)
        new_all_data = all_data.append(step_data)
        new_loop_state = (new_idx, new_all_data)
        chex.assert_trees_all_equal_shapes_and_dtypes(loop_state,new_loop_state)
        return new_loop_state
    
    end_loop_state = jax.lax.fori_loop(0,traj.length,step,init_loop_state)
    return end_loop_state[1]
