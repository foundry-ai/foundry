from stanza.envs import Environment
from typing import Any
from stanza.envs.goal_conditioned_envs.gc_pendulum import make_gc_pendulum_env

def create_gc(env : Environment, gs_sampler : Any,  env_name : str = 'pendulum'):
    if env_name == 'pendulum':
        return  make_gc_pendulum_env(env, gs_sampler)