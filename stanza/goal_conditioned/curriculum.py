from typing import Any, Callable, List
import jax
from jax.random import PRNGKey
from jax import random
from jax import numpy as jnp
from stanza.data import Data
from stanza.envs import Environment
from stanza.dataclasses import dataclass, replace, field
from stanza.envs.goal_conditioned_envs.gc_pendulum import make_gc_pendulum_env
from stanza.goal_conditioned import GCEnvironment
from stanza.goal_conditioned.roll_in_sampler import Noiser, RollInSampler
from stanza.rl.ppo import PPO
from stanza.train import Trainer
import optax
from stanza.rl.nets import MLPActorCritic
from stanza.rl import EpisodicEnvironment

# a mixture of environments
# each sampled with a given probability
# for resets
# transitions 



ScalarNoiserMaker = Callable[[float],Noiser]
ScaleFunction = Callable[[int],float]
# epoch, env

CurriculumInfo = Any


# runs ppo from inital params, a net, 
# and an environment, episode length, and optimizer
# TODO test this
def ppo_train(rng_key : PRNGKey, env : Environment, episode_length, net, 
              init_params,
              optimizer = optax.chain(
                        optax.clip_by_global_norm(0.5),
                        optax.adam(3e-4, eps=1e-5)
                    ), 
           init_opt_state = None,
            rl_hooks=[], train_hooks=[]):
    
    trainer_key, new_key = rng_key.split()
    ppo = PPO(
                trainer = Trainer(
                    optimizer=optimizer
                )
            )
    env = EpisodicEnvironment(env=env, episode_length=episode_length)
    trained_params = ppo.train(rng_key=trainer_key, 
            env=env, 
            actor_critic_apply=net.apply, 
            init_params=init_params, init_opt_state=None,
            rl_hooks=[], train_hooks=[])
            
    return trained_params, new_key
    


class ScheduleItem:
    schedule_env : Environment 
    optimizer : Any = optax.chain(
                        optax.clip_by_global_norm(0.5),
                        optax.adam(3e-4, eps=1e-5)
                    )
    episode_length : int = 1000



class ScheduleItemMaker:
    def make_schedule_item(epoch_num : int,
                                data : Data,
                                last_schedule_item : ScheduleItem,
                                curriculum_info : CurriculumInfo = None) -> ScheduleItem:
        raise NotImplementedError


    
class TrainingSchedule:
    init_params : Any
    num_epochs : int
    #bc_settings : Any = None
    episode_length : int = 1000
    base_env : Environment = None
    init_data : Data = None
    schedule_item_maker : ScheduleItemMaker = None

    ### In future, we can use data from earlier in curriculum
    ### to perform roll_in_sampling


    def execute_curriculum(self, init_key : PRNGKey = 42,
                           trainer_key : PRNGKey = 43):
        params = self.init_params
        base_env = self.base_env
        params_list = [params]

        # initialize the network
        net = MLPActorCritic(
            base_env.sample_action(PRNGKey(0))
            )
        params = net.init(init_key,
                base_env.observe(base_env.sample_state(PRNGKey(0))))


        for i in range(self.num_epochs):
            # make the schedule item
            schedule_item = self.schedule_item_maker.make_schedule_item(epoch_num=i,
                                                                        data=self.init_data,
                                                                        last_schedule_item=None)
            # train on the schedule item
            trained_params, trainer_key = ppo_train(rng_key=trainer_key, env=schedule_item.env, 
                                                    net=net, episode_length=schedule_item.episode_length, 
                                                    optimizer = schedule_item.optimizer, 
                init_params=params, init_opt_state = None,
                rl_hooks=[], train_hooks=[])
            
            params_list.append(trained_params)
            params = trained_params
            # update the schedule item maker
            


