import jax
from stanza.policies import Policy
from stanza.policies.transforms import PolicyTransform, PolicyInput, PolicyOutput
from typing import Any, Callable
from stanza.dataclasses import dataclass, replace
from stanza.goal_conditioned import GCState, GCObs
from jax.random import PRNGKey, split
Goal = Any

#GF stands for "GoalFix"

@dataclass(jax = True)
class GFPolicy(Policy):
    gc_policy : Policy
    goal : Goal 

    def __call__(self, input):
        new_obs = GCObs(input.observation, self.goal)
        new_input = replace(input,observation=new_obs)
        return self.gc_policy(new_input)

Obs = Any
PolicyState = Any

    

@dataclass(jax=True)
class BLPolicyState:
    chunk_time : int # time running LL policy
    current_goal : Goal #
    state_low_level : Any 
    state_high_level : Any


@dataclass(jax=True)
class BiPolicy(Policy):
    policy_low : Policy # is goal conditioned
    policy_high : Policy
    is_update_time : Callable[[BLPolicyState],bool] \
          = lambda x: x.chunk_time == 1

    def compute_goal(self,input,hl_rng_key):
        policy_state = input.policy_state
        hl_input = PolicyInput(
            observation=input.observation,
            rng_key=hl_rng_key,
            policy_state=policy_state.state_high_level \
            if policy_state is not None else None
        )
        hl_output = self.policy_high(hl_input)
        goal = hl_output.action    
        return goal, hl_output.policy_state
        
    def __call__(self, input):
        ll_rng_key, hl_rng_key = jax.random.split(input.rng_key)
        policy_state = input.policy_state
        if policy_state is None:
            goal, hl_policy_state = self.compute_goal(input,hl_rng_key)
            t = 0
        else: 
            t, (goal, hl_policy_state) = jax.lax.cond(
                self.is_update_time(policy_state),
                lambda: ( 0, self.compute_goal(input,hl_rng_key)),
                lambda: ( policy_state.chunk_time,
                         (policy_state.current_goal,
                          policy_state.state_high_level)),
                operand=None)
        # make the goal conditioned input
        # to the low level policy
        ll_input = PolicyInput(
            observation=GCObs(input.observation, goal),
            rng_key=ll_rng_key,
            policy_state=policy_state.state_low_level \
            if policy_state is not None else None
        )
        ll_output = self.policy_low(ll_input)
        new_policy_state = BLPolicyState(
            chunk_time=t + 1,
            current_goal=goal,
            state_low_level=ll_output.policy_state,
            state_high_level=hl_policy_state)
        
        return PolicyOutput(
            action=ll_output.action,
            policy_state=new_policy_state)
    
        