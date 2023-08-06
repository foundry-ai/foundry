

import jax
import jax.numpy as jnp
from stanza import policies
from stanza.data import Data
from stanza.data.trajectory import Timestep
from stanza.dataclasses import dataclass, field
from stanza.goal_conditioned.noisers import Noiser, ProportionFunction1d, \
NoiserGetter, even_proportions1d, make_gaussian_noiser_getter,\
    sample_from_prop_fn, none_getter
from stanza.policies import Policy
from jax.random import PRNGKey
from typing import Callable, Any, List
from stanza.goal_conditioned.exploration_policy import ToNoisePhaseFn, StateLoggingPolicy, SequentialNoisingPolicyState, SequentialNoisingPolicy

@dataclass(jax=True)
class SequentialNoisingPolicyBuilder:
    wrap_in_logging_policy : field(default=True,jax_static=True)


@dataclass(jax=True)
class RolloutCollectorFromSettings:
    base_policy : Policy
    num_settings : int = 1
    prop_function_noise: ProportionFunction1d = even_proportions1d
    prop_function_phase: ProportionFunction1d = even_proportions1d
    phase_noise_indepdent : bool = field(default=True,jax_static=True)
    
    noiser_getter : Callable[[int],Noiser] = None
    phase_fn_getter : Callable[[int],ToNoisePhaseFn] = None


    def make_logging_noising_policy(self, rng_key : PRNGKey, policy, noiser_getter):
        key1, key2, key3 = jax.random.split(rng_key,3)

    def get_logging_noising_policy(self, rng_key : PRNGKey):
        key1, key2 = jax.random.split(rng_key)
        
        noise_setting = sample_from_prop_fn(key1, self.prop_function_noise, self.num_settings)
        phase_setting = sample_from_prop_fn(key2, self.prop_function_phase, self.num_settings) \
            if self.phase_noise_indepdent else noise_setting
        
        noiser = self.noiser_getter(noise_setting) if self.noiser_getter \
              is not None else None
        phase_fn = self.phase_fn_getter(phase_setting) if self.noiser_getter \
                is not None else None
        
        noising_policy = SequentialNoisingPolicy(base_policy=self.base_policy, action_noiser = noiser,
                        to_noise_phase=phase_fn)
        return StateLoggingPolicy(base_policy=noising_policy)

    
    def rollout_policy(self,rng_key,env,horizon):
    # random init angle and angular velocity
        key1,key2,key3,key4 = jax.random.split(rng_key,4)
        x_0 = env.reset(key1) 
        pol = self.get_logging_noising_policy(key2)
        roll = policies.rollout(model = env.step,
                        state0 = x_0,
                        policy = pol,
                        length = horizon,
                        last_state = False,
                        policy_rng_key=key3,
                        model_rng_key=key4)

        #TODO @dan is this correct
        return Data.from_pytree(Timestep(roll.states,roll.actions,info=roll.info.pol_state))
        
        
    def batch_roll(self,rng_key, num_t,env,horizon):
        roll_func = jax.vmap(self.rollout_policy,in_axes=(0,None,None))
        rng_keys = jax.random.split(rng_key,num_t)
        return Data.from_pytree(roll_func(rng_keys,env,horizon))


        
