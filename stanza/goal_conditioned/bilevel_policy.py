import jax
from stanza.policies import Policy
from stanza.policies.transforms import PolicyTransform, PolicyInput, PolicyOutput
from typing import Any, Callable
from stanza.dataclasses import dataclass, replace, field
from stanza.goal_conditioned import GCState, GCObs
from jax.random import PRNGKey, split
from stanza.util.attrdict import AttrMap
from stanza import Partial

Goal = Any

#GF stands for "GoalFix"

Obs = Any
PolicyState = Any

    

@dataclass(jax=True)
class BLPolicyState:
    state_low_level : Any  = None
    state_high_level : Any = None
    chunk_time : int = 0# time running LL policy
    current_goal : Goal = None#
    info_high_level : Any = None


def always_update(state : BLPolicyState):
    return True 

# do i need to jit this?
def fixed_time_update(state : BLPolicyState, t_max = 2):
    return state.chunk_time > t_max


#TODO add default constructor things?
@dataclass(jax=True)
class BiPolicy(Policy):
    policy_low : Policy # is goal conditioned
    policy_high : Policy
    t_max : int = 2
    is_update_time : Callable[[BLPolicyState],bool] \
          = Partial(fixed_time_update,t_max=1)
    split_keys : bool = field(default=True,jax_static=True)
    #turn this off to aid in debuggin

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
        return goal, hl_output.policy_state, hl_output.info
        
    def __call__(self, input):
        ll_rng_key, hl_rng_key = jax.random.split(input.rng_key) if \
                                    self.split_keys else (input.rng_key,input.rng_key)


        policy_state = input.policy_state
        
        def get_hl_info():
            hl_info = input.policy_state.info_high_level
            if hl_info is not None:
                return hl_info
            return field(default_factory=AttrMap)

        if policy_state is None:
            goal, hl_policy_state, hl_info = self.compute_goal(input,hl_rng_key)
            t = 0
        else: 
            t, (goal, hl_policy_state, hl_info) = jax.lax.cond(
                self.is_update_time(policy_state),
                lambda x: ( 0, self.compute_goal(input,hl_rng_key)),
                lambda x: ( policy_state.chunk_time,
                         (policy_state.current_goal,
                          policy_state.state_high_level,
                          get_hl_info())),
                operand=None)
        # make the goal conditioned input
        # to the low level policy
        ll_input = PolicyInput(
            observation=GCObs(env_obs = input.observation, goal = goal),
            rng_key=ll_rng_key,
            policy_state=policy_state.state_low_level \
            if policy_state is not None else None
        )
        ll_output = self.policy_low(ll_input)
    

        new_policy_state = BLPolicyState(
            chunk_time=t + 1,
            current_goal=goal,
            state_low_level=ll_output.policy_state,
            state_high_level=hl_policy_state,
            info_high_level = hl_info)
        new_policy_info = AttrMap(high=hl_info, low = ll_output.info)

        return PolicyOutput(
            action=ll_output.action,
            policy_state=new_policy_state,
            info = new_policy_info)



def extract_high_output(output: PolicyOutput, use_goal_as_action: bool=True):
    action = jax.lax.cond(use_goal_as_action,
                    lambda: (output.BLPolicyState.current_goal),
                    lambda: (output.action),
                    operand=None)
    return PolicyOutput(action = action,
                    policy_state = output.policy_state.state_high_level,
                        info = output.info.high)

def extract_low_output(output: PolicyOutput):
    action = output.action
    return PolicyOutput(action = action,
                    policy_state = output.policy_state.state_low_level,
                        info = output.info.low)


# @ Dan? make this static?
class IdentityPolicy(Policy):
    is_gc : bool = True
    def __call__(self, input):
        action=input.observation
        if self.is_gc:
            action = action.goal
        return PolicyOutput(action = action,
            policy_state=input.policy_state)



def make_trivial_bi_policy(policy, t_max = -1):
    return BiPolicy(policy_low=IdentityPolicy(), 
                    policy_high = policy,
                    split_keys = False, t_max=t_max)
