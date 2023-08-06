from stanza.dataclasses import dataclass
from stanza.goal_conditioned.noisers import Noiser
from stanza.policies import Policy, PolicyInput, PolicyOutput
from typing import Callable, List, Any
from stanza.policies.transforms import PolicyTransform
from stanza.util.attrdict import AttrMap
import jax
from stanza.goal_conditioned.noisers import make_gaussian_noiser_scalar_variance


# ---- LoggingPolicy ----
class StateLoggingPolicy(Policy):
    base_policy: Callable

    def __call__(self,input):
        pol_output = self.base_policy(input)
        new_log = AttrMap(pol_state=pol_output.policy_state, pol_info=pol_output.info,
                          logging_format=True)
        return PolicyOutput(action=pol_output.action, policy_state=pol_output.policy_state,
                            info=new_log)


@dataclass(jax=True)
class SequentialNoisingPolicyState:
    base_state : Any = None
    phase_step : int = 0
    # current phase is a noise phase
    noise_phase : bool = False
    # laste step was a noise_phase
    last_noise_phase : bool = False
    #Tally's the number of COMPLETED noise cycles
    noise_cycles : int = 0

ToNoisePhaseFn = Callable[[SequentialNoisingPolicyState],bool]


@dataclass(jax=True)
class SequentialNoisingPolicy(Policy):
    base_policy : Policy
    action_noiser : Noiser = None
    to_noise_phase: ToNoisePhaseFn = None


    # key overrides input.rng_key
    def call_base(self,input,key = None):
        if input is None:
            return self.base_policy(None)
        
        base_state = input.policy_state.base_state if input.policy_state \
            is not None else None
        
        rng_key = input.rng_key if key is None else key
        base_input = PolicyInput(observation=input.observation, 
                                 policy_state=base_state, 
                                 rng_key=input.rng_key)
        return self.base_policy(base_input)

    def __call__(self, input):
        if input is None or input.policy_state is None:
            base_output = self.call_base(input)
            next_state = SequentialNoisingPolicyState(base_state = base_output.policy_state)
            action = base_output.action
        else:
            # split the rng key for policy and action
            rng_key, noise_key = jax.random.split(input.rng_key)
            base_output = self.call_base(input, key = rng_key)

            # noises the action if a noise_phase and non-null action noiser
            # not that even if action_noiser=None, it still splits the rng
            action = jax.lax.cond(input.policy_state.noise_phase and self.action_noiser is not None, 
                                  lambda x: self.action_noiser(noise_key, x, 
                                    input.policy_state.phase_step), 
                                    lambda x: x, operand = base_output.action)
            
            last_noise_phase = input.policy_state.last_noise_phase
            # determines if a next noise phase
            next_noise_phase = self.to_noise_phase(input.policy_state) if self.to_noise_phase is not None else False

            # augments noise cycle if switching from noise_phase to next_noise_phase
            noise_cycles = jax.lax.cond(input.policy_state.noise_phase and not next_noise_phase 
                                            ,lambda x: x + 1, lambda x: x, operand = input.policy_state.noise_cycles)
            
            #augments counter if staying in next noise phase
            phase_step = jax.lax.cond(input.policy_state.noise_phase==next_noise_phase, lambda x: x + 1, lambda x: x, 
                                    operand = input.policy_state.phase_step)
            
            next_state = SequentialNoisingPolicyState(base_state=base_output.policy_state,
                                                        noise_phase=next_noise_phase,
                                                        phase_step=phase_step,
                                                        noise_cycles=noise_cycles,
                                                        last_noise_phase=last_noise_phase)
        
        return PolicyOutput(action=action, policy_state=next_state, info=base_output.info)


def make_to_timing_to_noise_phase(noise_step : int = 5, clean_step : int = 5):
    def to_noise_phase(state : SequentialNoisingPolicyState):
        if state is None:
            return False

        keep_noising = state.noise_phase & (state.phase_step < noise_step)
        switch_to_noising = (not state.not_phase) & (state.phase_step >= clean_step)

        return keep_noising or switch_to_noising
    return to_noise_phase

def make_timed_gaussian_sequential_noising_policy(policy : Policy,
                                                  sigma : float = 0, 
                                                  noise_step : int = 5,
                                                  clean_step : int = 5):
    to_noise_phase = make_to_timing_to_noise_phase(noise_step=noise_step, clean_step=clean_step)
    action_noiser = make_gaussian_noiser_scalar_variance(sigma)
    return SequentialNoisingPolicy(base_policy=policy, action_noiser=action_noiser,
                                      to_noise_phase=to_noise_phase)

