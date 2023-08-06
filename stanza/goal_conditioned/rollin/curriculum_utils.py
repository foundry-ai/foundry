from stanza.dataclasses import dataclass, replace, field
from stanza.envs import Environment
import jax.numpy as jnp
import jax
import jax.random as random
from stanza.goal_conditioned.rollin.curriculum import ScheduleItem, ScheduleItemMaker, CurriculumInfo
from typing import Any, Callable, List
from stanza.goal_conditioned import GCEnvironment
from stanza.goal_conditioned.rollin.rollin_sampler import Noiser, RollInHelper, RollInSampler
from stanza.data import Data
from jax.tree_util import Partial  
from stanza.goal_conditioned.noisers import ProportionFunction1d,ProportionFunction2d,NoiserGetter, even_proportions2d

"""
ToDo: Allow schedule Item to vary the rollin lengths
"""

"""
Makes the Mixture Environemnts
"""


# first int is the epoch, second is the index of the env
EnvMaker = Callable[[Environment,Any],GCEnvironment]
RollinHelperGetter = Callable[[int],RollInHelper]



@dataclass(jax=True)
class MixtureEnvironment(Environment):
    envs_list : List 
    first_is_base : bool = field(default=True,jax_static=True)
    mixture_base_env : Environment = None

    #TODO type this appropriately
    probs_list : jnp.ndarray = None 
    
    #split_in_sample : field(default=True,jax_static=True)


    def sample_env(self, rng_key):
        key1, key2 = random.split(rng_key)
        probs = jnp.array([1 for _ in range(len(self.envs_list))]) \
            if self.probs_list \
            is None else self.probs_list
        #print(probs)
        env_ind = random.choice(key1, len(self.envs_list), p = probs)
        env = self.envs_list[env_ind]
        return (env, key2, env_ind)
    
    def get_mixture_base_env(self):
        if self.mixture_base_env is None:
            if self.first_is_base:
                return self.envs_list[0]
            else:
                # TODO: make this sampling use key 
                self.sample_env(random.PRNGKey(0))[0]
        else:
            return self.mixture_base_env

    def sample_state_at_envnum(self, rng_key, env_num):
        return self.envs_list[env_num].sample_state(rng_key)
    
    def sample_action_at_envnum(self, rng_key, env_num):
        return self.envs_list[env_num].sample_action(rng_key)


    def sample_state(self, rng_key):
        (env,key,_) = self.sample_env(rng_key)
        return env.sample_state(key)

    def sample_action(self, rng_key):
        (env,key,_) = self.sample_env(rng_key)
        return env.sample_action(key)

    def reset(self, key):
        (env,key,_) = self.sample_env(key)
        return env.reset(key)  

    # these methods use the mixture_base_env 
    def step(self, state, action, rng_key):
        return self.get_mixture_base_env().step(state, action, rng_key)
        

    def observe(self, state):
        return self.get_mixture_base_env().observe(state)
    
    def reward(self, state, action, next_state):
        return self.get_mixture_base_env().reward(state, action, next_state)
    
    def render(self,state,**kwargs):
        return self.get_mixture_base_env().render(state.env_state, **kwargs)

    # this takes advantage of the fact that
    # probabilities need not be weighted
    def append_to_mixture(self,next_env, next_prob):
        new_envs = self.envs_list + [next_env]
        new_probs = jnp.append(self.probs_list, next_prob)
        #is this correct?
        return self.append_env_to_mixture_with_prob(new_envs,new_probs)
    
    def append_env_to_mixture_with_prob(self, next_env, new_probs):
        new_envs = self.envs_list + [next_env]
        return replace(self, envs_list=new_envs, probs_list=new_probs)

    def append_env_to_mixture_with_prob_fn(self, next_env, prob_fn):
        new_envs = self.envs_list + [next_env]
        new_probs = jax.vmap(prob_fn, in_axes=(0))(jnp.arange(len(new_envs)))
        return replace(self, envs_list=new_envs, probs_list=new_probs)
# assumes schedule environments are mixtures

class GCScheduleItemMaker(ScheduleItemMaker):
    base_env : Environment 
    gc_env_maker : Any 
    gc_base_env : Environment = None
    action_noiser_getter : NoiserGetter = None
    process_noiser_getter : NoiserGetter = None
    prob_fn  : ProportionFunction2d = even_proportions2d
    rollin_helper_getter : RollinHelperGetter = None
    zero_init_noise : bool = field(default=False,jax_static=True)
   
    def make_schedule_item(self,epoch_num : int,
                                data : Data,
                                last_schedule_item : ScheduleItem,
                                curriculum_info : CurriculumInfo = None):
        

        # sets up the noisers and roll_in_helps
        # none values accepted 
        action_noiser = self.action_noiser_getter(epoch_num) \
            if self.action_noiser_getter is not None else None
        process_noiser = self.process_noiser_getter(epoch_num)  \
            if self.process_noiser_getter is not None else None
        roll_in_helper = self.rollin_helper_getter(epoch_num) \
            if self.rollin_helper_getter is not None else None
        
        
        # forces zero noising at the start
        if self.zero_init_noise and (epoch_num == 0 or last_schedule_item is None):
            action_noiser, process_noiser = None, None

        # creates the next gc_env to add to the mixture 
        supplied_args = {'env':self.base_env, 'traj_data':data,
                          'action_noiser':action_noiser, 
                          'process_noiser':process_noiser, 'roll_in_helper':roll_in_helper}

        supplied_args = {k:v for k,v in supplied_args.items() if v is not None}
        sampler = RollInSampler(**supplied_args)
        gs_sampler = (lambda key: sampler.sample_gc_state(key))
        next_env = self.gc_env_maker(env=self.base_env, gs_sampler=gs_sampler)

        # if starting from scratch,
        # set up the first schedule item
        if last_schedule_item is None or\
            last_schedule_item.schedule_env is None \
            or epoch_num == 0:

            probs_list = jnp.array(1)
            mix_env = MixtureEnvironment(envs_list=[next_env], 
                                         mixture_base_env=self.gc_base_env)
            return ScheduleItem(env=mix_env)
        
        mixture_env = last_schedule_item.schedule_env
                                                                    
        #next_mix_env = mixture_env.append_env_to_mixture_with_prob(next_env, probs_list)
        # first int is the epoch, second is the index of the env
        epoch_prob_fn = lambda i : self.prob_fn(epoch_num,i)
        next_mix_env = mixture_env.append_env_to_mixture_with_prob_fn(next_env, epoch_prob_fn)

        return ScheduleItem(env=next_mix_env)