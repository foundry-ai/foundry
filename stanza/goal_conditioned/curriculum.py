from typing import Any
import jax
from jax.random import PRNGKey
from jax import random
from jax import numpy as jnp
from stanza.envs import Environment
from stanza.dataclasses import dataclass, replace, field
from stanza.rl.ppo import PPO
from stanza.train import Trainer
import optax
from stanza.rl.nets import MLPActorCritic
from stanza.rl import EpisodicEnvironment

# a mixture of environments
# each sampled with a given probability
# for resets
# transitions 
@dataclass(jax=True)
class MixtureEnvironment(Environment):

    envs_list : list = []
    probs_list : list = []
    split_in_sample : field(default=True,jax_static=True)
    base_env : Environment = None

    def sample_env(self, rng_key):
        key1, key2 = random.split(rng_key)
        env, env_str = random.choice(key1, self.envs_list, p = self.probs_list)
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
    env : MixtureEnvironment # need a discrete distribution
    optimizer : Any = optax.chain(
                        optax.clip_by_global_norm(0.5),
                        optax.adam(3e-4, eps=1e-5)
                    )
    episode_length : int = 1000

class TrainingSchedule:
    init_params : Any
    schedule : list = []
    bc_settings : Any = None
    episode_length : int = 1000
    base_env : Environment = None
    

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


        if self.bc_data is not None:
            pass
            # train bc model
        for item in self.schedule:
            trained_params, trainer_key = ppo_train(rng_key=trainer_key, env=item.env, 
                                                    net=net, episode_length=item.episode_length, 
                                                    optimizer = item.optimizer, 
                init_params=params, init_opt_state = None,
                rl_hooks=[], train_hooks=[])
            params_list.append(trained_params)
            

        
        
