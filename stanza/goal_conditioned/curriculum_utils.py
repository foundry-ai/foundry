from stanza.dataclasses import dataclass, replace, field
from stanza.envs import Environment
import jax.numpy as jnp
import jax
import jax.random as random
from stanza.goal_conditioned.curriculum import ScheduleItem, ScheduleItemMaker, CurriculumInfo
from typing import Any, Callable, List
from stanza.goal_conditioned import GCEnvironment
from stanza.goal_conditioned.roll_in_sampler import Noiser, RollInHelper, RollInSampler
from stanza.data import Data
from jax.tree_util import Partial  
from stanza.distribution.mvn import MultivariateNormalDiag


"""
ToDo: Allow schedule Item to vary the rollin lengths
"""

"""
Makes the Mixture Environemnts
"""


ProportionFunction = Callable[[int,int],float]
EnvMaker = Callable[[Environment,Any],GCEnvironment]
NoiserGetter = Callable[[int],Noiser]
RollinHelperGetter = Callable[[int],RollInHelper]



def even_proportions(i : int , j : int):
    return 1.



def make_gaussian_noiser_scalar_variance(sample, sigma):
    """
    @param sample : a function that takes a PRNGKey and returns a sample
    @param scale_diag : a scalar
    @return a noiser that takes a PRNGKey and a value and returns a noised value
    """
    #sample_flat, sample_uf = jax.flatten_util.ravel_pytree(sample)
    #zero_flat = jnp.zeros(sample.shape[0],)
    #var_flat = jnp.ones(sample.shape[0],) * sigma
    #mvn_dist = MultivariateNormalDiag(sample_uf(zero_flat), sample_uf(var_flat))
    def noiser(rng, value):
        sigs = jax.tree_map(lambda x: sigma*jnp.ones_like(x), value)
        mvn = MultivariateNormalDiag(value, sigs)
        return mvn.sample(rng)

        noise = mvn_dist.sample(rng)
        noise_flat, _ = jax.flatten_util.ravel_pytree(noise)
        value_flat, _ = jax.flatten_util.ravel_pytree(value)
        return sample_uf(value_flat + noise_flat)
    return noiser


"""
creates_gaussian noiser of shape @sample
using a function to specialize the scedule 
"""

def make_gaussian_noiser_getter(var_fn : Callable[[int],float],sample):
    def noiser_getter(epoch_num: int):
        return make_gaussian_noiser_scalar_variance(sample, var_fn(epoch_num))
    return noiser_getter


@dataclass(jax=True)
class MixtureEnvironment(Environment):

    envs_list : list = []
    probs_list : jax.Array = []
    split_in_sample : field(default=True,jax_static=True)
    base_env : Environment = None

    def sample_env(self, rng_key):
        key1, key2 = random.split(rng_key)
        env_ind = random.choice(key1, len(self.envs_list), p = self.probs_list)
        env = self.envs_list(env_ind)
        return (env, key2)
    
    def sample_state(self, rng_key):
        (env,key) = self.sample_env(rng_key)
        return env.sample_state(key)

    def sample_action(self, rng_key):
        (env,key) = self.sample_env(rng_key)
        return env.sample_action(key)

    def reset(self, key):
        (env,key) = self.sample_env(key)
        return env.reset(key)  

    # these methods use the base_env 
    def step(self, state, action, rng_key):
        return self.base_env(rng_key)

    def observe(self, state):
        return self.base_env.observe(state)
    
    def reward(self, state, action, next_state):
        return self.base_env.reward(state, action, next_state)
    
    def render(self,state,**kwargs):
        return self.base_env.render(state.env_state, **kwargs)



# assumes schedule environments are mixtures
class GCScheduleItemMaker(ScheduleItemMaker):
    env_maker : Any = None
    base_env : Environment = None
    action_noiser_getter : NoiserGetter = None
    process_noiser_getter : NoiserGetter = None
    prop_fn  : ProportionFunction = even_proportions
    rollin_helper_getter : RollinHelperGetter = None

   
    def make_schedule_item(self,epoch_num : int,
                                data : Data,
                                last_schedule_item : ScheduleItem,
                                curriculum_info : CurriculumInfo = None):
        
        
        if last_schedule_item is None or epoch_num is 0:
            sampler = RollInSampler(env=self.base_env, traj_data=data, action_noiser=action_noiser, process_noiser=process_noiser)
            gs_sampler = (lambda key: sampler.sample_gc_state(key))
            next_env = self.env_maker(env=self.base_env, gs_sampler=gs_sampler)
            envs_list = [next_env]
            probs_list = jnp.array(1)
            mix_env = MixtureEnvironment(envs_list=envs_list, probs_list=probs_list, base_env=self.base_env)
            return ScheduleItem(env=mix_env)
        
        mixture_env = last_schedule_item.schedule_env
        envs_list = mixture_env.envs_list
        
        
        action_noiser = self.action_noiser_getter(epoch_num) \
            if self.action_noiser_getter is not None else None
        process_noiser = self.process_noiser_getter(epoch_num)  \
            if self.process_noiser_getter is not None else None
        roll_in_helper = self.rollin_helper_getter(epoch_num) \
            if self.rollin_helper_getter is not None else None
        
        supplied_args = {'env':self.base_env, 'traj_data':data,
                          'action_noiser':action_noiser, 
                          'process_noiser':process_noiser, 'roll_in_helper':roll_in_helper}

        supplied_args = {k:v for k,v in supplied_args.items() if v is not None}
        sampler = RollInSampler(**supplied_args)
        gs_sampler = (lambda key: sampler.sample_gc_state(key))

        next_env = self.env_maker(env=self.base_env, gs_sampler=gs_sampler)
        envs_list = envs_list + [next_env]
        probs_list = jnp.array([self.prop_fn(epoch_num,j) for j in range(len(envs_list))])
        
        mix_env = MixtureEnvironment(envs_list=envs_list, probs_list=probs_list, base_env=self.base_env)
        
        return ScheduleItem(env=mix_env)